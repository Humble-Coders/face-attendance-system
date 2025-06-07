import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import base64
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import logging
import json
import time
import face_recognition

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add Silent Face Anti-Spoofing to path
sys.path.append('/app/silent-antispoofing')
sys.path.append('/app/silent-antispoofing/src')

app = Flask(__name__)
CORS(app)

# Initialize Firebase
db = None
try:
    # Try to read from environment variable first
    firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
    if firebase_creds:
        logger.info("Loading Firebase credentials from environment variable")
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
    else:
        # Fallback to file
        cred = credentials.Certificate('/app/firebase-credentials.json')
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {e}")
    db = None

# Initialize Anti-Spoofing Service
class AntiSpoofingService:
    def __init__(self):
        self.model_dir = '/app/silent-antispoofing/resources/anti_spoof_models'
        self.model_test = None
        self.image_cropper = None
        
        try:
            from anti_spoof_predict import AntiSpoofPredict
            from generate_patches import CropImage
            from utility import parse_model_name
            
            self.model_test = AntiSpoofPredict(0)  # CPU device
            self.image_cropper = CropImage()
            self.parse_model_name = parse_model_name
            logger.info("Anti-spoofing service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize anti-spoofing: {e}")
    
    def predict(self, image):
        """Predict if face is real or fake"""
        if self.model_test is None:
            logger.warning("Anti-spoofing not available, using fallback")
            return self.basic_liveness_check(image)
        
        try:
            height, width, _ = image.shape
            bbox = [0, 0, width, height]
            
            prediction = np.zeros((1, 3))
            
            # Check if model files exist
            model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pth')]
            if not model_files:
                logger.warning("No model files found, using basic check")
                return self.basic_liveness_check(image)
            
            for model_name in model_files:
                h_input, w_input, model_type, scale = self.parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                
                img = self.image_cropper.crop(**param)
                pred = self.model_test.predict(img, os.path.join(self.model_dir, model_name))
                prediction += pred
            
            # Calculate final score
            label = np.argmax(prediction)
            value = prediction[0][label] / len(model_files)
            
            logger.info(f"Anti-spoofing prediction: {value}")
            return float(value)
            
        except Exception as e:
            logger.error(f"Anti-spoofing prediction failed: {e}")
            return self.basic_liveness_check(image)
    
    def basic_liveness_check(self, image):
        """Basic fallback liveness check"""
        try:
            height, width = image.shape[:2]
            
            # Check image size
            if height < 100 or width < 100:
                return 0.2
            
            # Check brightness distribution
            mean_brightness = np.mean(image)
            if mean_brightness < 30 or mean_brightness > 220:
                return 0.3
            
            # Check edge density (fake images often have different edge characteristics)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            if edge_density < 0.01 or edge_density > 0.3:
                return 0.4
            
            return 0.75  # Reasonable score for basic check
            
        except Exception as e:
            logger.error(f"Basic liveness check failed: {e}")
            return 0.1

# Initialize services
anti_spoof_service = AntiSpoofingService()

# Face Recognition Service using face_recognition library
class FaceRecognitionService:
    def __init__(self):
        self.known_face_encodings = {}
        self.load_known_faces()
    
    def load_known_faces(self):
        """Load known faces from Firebase"""
        if not db:
            return
        
        try:
            students_ref = db.collection('students')
            docs = students_ref.stream()
            
            for doc in docs:
                student_data = doc.to_dict()
                if student_data.get('faceEncoding'):
                    encoding = np.array(student_data['faceEncoding'])
                    self.known_face_encodings[student_data['studentId']] = encoding
            
            logger.info(f"Loaded {len(self.known_face_encodings)} known faces")
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
    
    def encode_face(self, image):
        """Extract face encoding from image"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                return None
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if face_encodings:
                return face_encodings[0]
            
            return None
        except Exception as e:
            logger.error(f"Face encoding failed: {e}")
            return None
    
    def add_known_face(self, student_id, image):
        """Add a new face to known faces"""
        encoding = self.encode_face(image)
        if encoding is not None:
            self.known_face_encodings[student_id] = encoding
            return encoding.tolist()  # Convert to list for JSON storage
        return None
    
    def recognize_face(self, image, tolerance=0.6):
        """Recognize face in image"""
        try:
            encoding = self.encode_face(image)
            if encoding is None:
                return None, 0.0
            
            if not self.known_face_encodings:
                return None, 0.0
            
            # Compare with known faces
            student_ids = list(self.known_face_encodings.keys())
            known_encodings = list(self.known_face_encodings.values())
            
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=tolerance)
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            
            if True in matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    confidence = 1 - face_distances[best_match_index]
                    return student_ids[best_match_index], confidence
            
            return None, 0.0
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0

# Initialize face recognition service
face_recognition_service = FaceRecognitionService()

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return opencv_image
    except Exception as e:
        logger.error(f"Failed to convert base64 to image: {e}")
        return None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "Face Recognition Attendance System",
        "version": "1.0.0",
        "status": "running",
        "services": {
            "firebase": db is not None,
            "anti_spoofing": anti_spoof_service.model_test is not None,
            "face_recognition": True,
            "known_faces": len(face_recognition_service.known_face_encodings)
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "firebase_connected": db is not None,
        "anti_spoofing_ready": anti_spoof_service.model_test is not None,
        "face_recognition_ready": True,
        "known_faces_count": len(face_recognition_service.known_face_encodings)
    })

@app.route('/register-student', methods=['POST'])
def register_student():
    try:
        data = request.json
        student_id = data.get('studentId')
        name = data.get('name')
        face_image_base64 = data.get('faceImage')
        
        logger.info(f"Registration request for student: {student_id}")
        
        if not all([student_id, name, face_image_base64]):
            return jsonify({"error": "Missing required fields: studentId, name, faceImage"}), 400
        
        # Convert base64 to image
        image = base64_to_image(face_image_base64)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Check liveness
        liveness_score = anti_spoof_service.predict(image)
        logger.info(f"Liveness score for {student_id}: {liveness_score}")
        
        if liveness_score < 0.5:
            return jsonify({
                "error": "Spoof detected during registration. Please use a real face.",
                "liveness_score": liveness_score
            }), 400
        
        # Extract face encoding
        face_encoding = face_recognition_service.add_known_face(student_id, image)
        if face_encoding is None:
            return jsonify({"error": "No face detected in image"}), 400
        
        # Save to Firebase
        if db:
            # Check if student already exists
            existing_student = db.collection('students').document(student_id).get()
            if existing_student.exists:
                return jsonify({"error": "Student already registered"}), 400
            
            student_data = {
                'studentId': student_id,
                'name': name,
                'faceEncoding': face_encoding,
                'livenessScore': liveness_score,
                'registeredAt': datetime.now(),
                'isActive': True
            }
            db.collection('students').document(student_id).set(student_data)
            logger.info(f"Student {student_id} registered successfully")
        
        return jsonify({
            "success": True,
            "message": f"Student {name} registered successfully",
            "studentId": student_id,
            "liveness_score": liveness_score
        })
        
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return jsonify({"error": f"Registration failed: {str(e)}"}), 500

@app.route('/mark-attendance', methods=['POST'])
def mark_attendance():
    try:
        data = request.json
        student_id = data.get('studentId')
        face_image_base64 = data.get('faceImage')
        
        logger.info(f"Attendance request for student: {student_id}")
        
        if not all([student_id, face_image_base64]):
            return jsonify({"error": "Missing required fields: studentId, faceImage"}), 400
        
        # Convert base64 to image
        image = base64_to_image(face_image_base64)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Check liveness
        liveness_score = anti_spoof_service.predict(image)
        logger.info(f"Liveness score for {student_id}: {liveness_score}")
        
        if liveness_score < 0.5:
            return jsonify({
                "error": "Spoof detected. Please use your real face.",
                "liveness_score": liveness_score
            }), 400
        
        # Recognize face
        recognized_id, confidence = face_recognition_service.recognize_face(image)
        logger.info(f"Recognition result: {recognized_id}, confidence: {confidence}")
        
        # Check if recognized face matches provided student ID
        if recognized_id != student_id or confidence < 0.6:
            return jsonify({
                "error": "Face does not match the provided Student ID",
                "liveness_score": liveness_score,
                "recognition_confidence": confidence,
                "recognized_student": recognized_id
            }), 400
        
        # Check if student exists and mark attendance
        if db:
            student_ref = db.collection('students').document(student_id)
            student_doc = student_ref.get()
            
            if not student_doc.exists:
                return jsonify({"error": "Student not found in database"}), 404
            
            student_data = student_doc.to_dict()
            
            # Check if already marked today
            today = datetime.now().date().isoformat()
            existing_attendance = list(db.collection('attendance')
                .where('studentId', '==', student_id)
                .where('date', '==', today)
                .limit(1).stream())
            
            if len(existing_attendance) > 0:
                return jsonify({"error": "Attendance already marked today"}), 400
            
            # Mark attendance
            attendance_data = {
                'studentId': student_id,
                'studentName': student_data.get('name'),
                'timestamp': datetime.now(),
                'date': today,
                'livenessScore': liveness_score,
                'recognitionConfidence': confidence,
                'status': 'present',
                'markedAt': datetime.now().isoformat()
            }
            
            attendance_id = f"{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            db.collection('attendance').document(attendance_id).set(attendance_data)
            
            logger.info(f"Attendance marked for {student_id} with confidence {confidence}")
            
            return jsonify({
                "success": True,
                "message": f"Attendance marked for {student_data.get('name')}",
                "studentName": student_data.get('name'),
                "liveness_score": liveness_score,
                "recognition_confidence": confidence,
                "timestamp": attendance_data['markedAt']
            })
        
        return jsonify({"error": "Database not connected"}), 500
        
    except Exception as e:
        logger.error(f"Attendance marking failed: {e}")
        return jsonify({"error": f"Attendance marking failed: {str(e)}"}), 500

@app.route('/get-students', methods=['GET'])
def get_students():
    try:
        if not db:
            return jsonify({"error": "Database not connected"}), 500
        
        students = []
        students_ref = db.collection('students')
        docs = students_ref.stream()
        
        for doc in docs:
            student_data = doc.to_dict()
            # Don't return face encoding in the list
            students.append({
                'studentId': student_data.get('studentId'),
                'name': student_data.get('name'),
                'registeredAt': student_data.get('registeredAt'),
                'isActive': student_data.get('isActive', True)
            })
        
        return jsonify({"students": students, "count": len(students)})
        
    except Exception as e:
        logger.error(f"Failed to get students: {e}")
        return jsonify({"error": "Failed to retrieve students"}), 500

@app.route('/get-attendance', methods=['GET'])
def get_attendance():
    try:
        if not db:
            return jsonify({"error": "Database not connected"}), 500
        
        # Get query parameter for date (default to today)
        date_param = request.args.get('date', datetime.now().date().isoformat())
        
        attendance_records = []
        attendance_ref = db.collection('attendance').where('date', '==', date_param)
        docs = attendance_ref.stream()
        
        for doc in docs:
            attendance_data = doc.to_dict()
            attendance_records.append({
                'studentId': attendance_data.get('studentId'),
                'studentName': attendance_data.get('studentName'),
                'timestamp': attendance_data.get('markedAt'),
                'livenessScore': attendance_data.get('livenessScore'),
                'recognitionConfidence': attendance_data.get('recognitionConfidence'),
                'status': attendance_data.get('status')
            })
        
        return jsonify({
            "attendance": attendance_records,
            "count": len(attendance_records),
            "date": date_param
        })
        
    except Exception as e:
        logger.error(f"Failed to get attendance: {e}")
        return jsonify({"error": "Failed to retrieve attendance"}), 500

@app.route('/reload-faces', methods=['POST'])
def reload_faces():
    """Reload known faces from database"""
    try:
        face_recognition_service.load_known_faces()
        return jsonify({
            "success": True,
            "message": "Known faces reloaded",
            "count": len(face_recognition_service.known_face_encodings)
        })
    except Exception as e:
        logger.error(f"Failed to reload faces: {e}")
        return jsonify({"error": "Failed to reload faces"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
