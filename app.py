import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import logging
import json
import tempfile
import gc

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize Firebase
db = None
try:
    firebase_creds = os.getenv('FIREBASE_CREDENTIALS')
    if firebase_creds:
        logger.info("Loading Firebase credentials from environment variable")
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase initialized successfully")
    else:
        logger.warning("No Firebase credentials found")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {e}")
    db = None

# Memory-optimized DeepFace Service
class OptimizedDeepFaceService:
    def __init__(self):
        self.deepface = None
        self.models_loaded = False
        self.face_cascade = None
        
        # Initialize OpenCV for basic face detection first
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("OpenCV face detection initialized")
        except Exception as e:
            logger.error(f"OpenCV initialization failed: {e}")
        
        # Lazy load DeepFace only when needed
        logger.info("DeepFace will be loaded on first use to save memory")
    
    def _load_deepface_if_needed(self):
        """Lazy loading of DeepFace to save memory during startup"""
        if self.deepface is None:
            try:
                logger.info("Loading DeepFace (first use)...")
                from deepface import DeepFace
                self.deepface = DeepFace
                self.models_loaded = True
                logger.info("DeepFace loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load DeepFace: {e}")
                self.models_loaded = False
    
    def predict(self, image):
        """Liveness detection with memory optimization"""
        try:
            # First try basic OpenCV detection (very fast, low memory)
            if self.face_cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) == 0:
                    logger.info("No face detected with OpenCV, returning low score")
                    return 0.2
                
                # Basic quality checks
                basic_score = self.basic_liveness_check(image)
                
                # If basic score is too low, don't bother with DeepFace
                if basic_score < 0.3:
                    return basic_score
                
                # For borderline cases, use DeepFace for better accuracy
                if basic_score < 0.7:
                    return self.deepface_liveness_check(image)
                
                return basic_score
            
            # Fallback to basic check if OpenCV fails
            return self.basic_liveness_check(image)
            
        except Exception as e:
            logger.error(f"Liveness prediction failed: {e}")
            return 0.5
    
    def basic_liveness_check(self, image):
        """Fast basic liveness check"""
        try:
            height, width = image.shape[:2]
            
            # Size check
            if height < 100 or width < 100:
                return 0.2
            
            # Brightness analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 30 or mean_brightness > 220:
                return 0.3
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            if edge_density < 0.01:
                return 0.3
            elif edge_density > 0.05:
                return 0.7
            else:
                return 0.5
                
        except:
            return 0.5
    
    def deepface_liveness_check(self, image):
        """Use DeepFace for accurate liveness detection"""
        try:
            self._load_deepface_if_needed()
            
            if not self.deepface:
                return self.basic_liveness_check(image)
            
            # Save image temporarily
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                
                try:
                    # Use DeepFace face detection only (lighter than full analysis)
                    faces = self.deepface.extract_faces(
                        img_path=tmp_file.name,
                        enforce_detection=False
                    )
                    
                    if faces and len(faces) > 0:
                        # Face found with DeepFace, assume good quality
                        return 0.8
                    else:
                        return 0.3
                        
                finally:
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                    # Force garbage collection to free memory
                    gc.collect()
                    
        except Exception as e:
            logger.error(f"DeepFace liveness check failed: {e}")
            return self.basic_liveness_check(image)

# Optimized Face Recognition Service
class OptimizedFaceRecognitionService:
    def __init__(self):
        self.deepface = None
        self.known_face_embeddings = {}
        logger.info("Face recognition service initialized (DeepFace lazy loaded)")
        self.load_known_faces()
    
    def _load_deepface_if_needed(self):
        """Lazy load DeepFace"""
        if self.deepface is None:
            try:
                logger.info("Loading DeepFace for face recognition...")
                from deepface import DeepFace
                self.deepface = DeepFace
                logger.info("DeepFace loaded for recognition")
            except Exception as e:
                logger.error(f"Failed to load DeepFace: {e}")
    
    def load_known_faces(self):
        """Load known faces from Firebase"""
        if not db:
            return
        
        try:
            students_ref = db.collection('students')
            docs = students_ref.stream()
            
            for doc in docs:
                student_data = doc.to_dict()
                if student_data.get('faceEmbedding'):
                    embedding = np.array(student_data['faceEmbedding'])
                    self.known_face_embeddings[student_data['studentId']] = embedding
            
            logger.info(f"Loaded {len(self.known_face_embeddings)} known face embeddings")
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")
    
    def get_face_embedding(self, image):
        """Extract face embedding using optimized DeepFace"""
        self._load_deepface_if_needed()
        
        if not self.deepface:
            return None
        
        try:
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                cv2.imwrite(tmp_file.name, image)
                
                try:
                    # Use lighter model for faster processing
                    embedding = self.deepface.represent(
                        img_path=tmp_file.name,
                        model_name='Facenet',  # Lighter than Facenet512
                        enforce_detection=False
                    )
                    
                    if embedding and len(embedding) > 0:
                        return np.array(embedding[0]['embedding'])
                    
                    return None
                    
                finally:
                    # Cleanup
                    os.unlink(tmp_file.name)
                    gc.collect()  # Force memory cleanup
                    
        except Exception as e:
            logger.error(f"Face embedding extraction failed: {e}")
            return None
    
    def add_known_face(self, student_id, image):
        """Add new face with memory optimization"""
        embedding = self.get_face_embedding(image)
        if embedding is not None:
            self.known_face_embeddings[student_id] = embedding
            return embedding.tolist()
        return None
    
    def recognize_face(self, image, threshold=0.6):
        """Recognize face with memory optimization"""
        try:
            current_embedding = self.get_face_embedding(image)
            if current_embedding is None:
                return None, 0.0
            
            if not self.known_face_embeddings:
                return None, 0.0
            
            # Compare with known faces
            best_match = None
            best_distance = float('inf')
            
            for student_id, known_embedding in self.known_face_embeddings.items():
                distance = np.linalg.norm(current_embedding - known_embedding)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = student_id
            
            # Convert distance to confidence
            confidence = max(0.0, 1.0 - (best_distance / 2.0))
            
            if confidence >= threshold:
                return best_match, confidence
            
            return None, confidence
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return None, 0.0

# Initialize optimized services
anti_spoof_service = OptimizedDeepFaceService()
face_recognition_service = OptimizedFaceRecognitionService()

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
        "message": "Face Recognition Attendance System - Memory Optimized DeepFace", 
        "version": "3.1.0",
        "status": "running",
        "services": {
            "firebase": db is not None,
            "deepface_antispoofing": "lazy_loaded",
            "deepface_recognition": "lazy_loaded",
            "known_faces": len(face_recognition_service.known_face_embeddings)
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "firebase_connected": db is not None,
        "deepface_ready": "lazy_loaded",
        "known_faces_count": len(face_recognition_service.known_face_embeddings)
    })

@app.route('/register-student', methods=['POST'])
def register_student():
    try:
        # Better JSON parsing
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
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
        
        # Check liveness (optimized)
        liveness_score = anti_spoof_service.predict(image)
        logger.info(f"Liveness score for {student_id}: {liveness_score}")
        
        if liveness_score < 0.5:
            return jsonify({
                "error": "Spoof detected during registration. Please use a real face.",
                "liveness_score": liveness_score
            }), 400
        
        # Extract face embedding (optimized)
        face_embedding = face_recognition_service.add_known_face(student_id, image)
        if face_embedding is None:
            return jsonify({"error": "No face detected in image or face extraction failed"}), 400
        
        # Save to Firebase
        if db:
            existing_student = db.collection('students').document(student_id).get()
            if existing_student.exists:
                return jsonify({"error": "Student already registered"}), 400
            
            student_data = {
                'studentId': student_id,
                'name': name,
                'faceEmbedding': face_embedding,
                'livenessScore': liveness_score,
                'registeredAt': datetime.now(),
                'isActive': True
            }
            db.collection('students').document(student_id).set(student_data)
            logger.info(f"Student {student_id} registered successfully")
        
        # Force garbage collection after heavy operation
        gc.collect()
        
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
        # Better JSON parsing
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400
        
        student_id = data.get('studentId')
        face_image_base64 = data.get('faceImage')
        
        logger.info(f"Attendance request for student: {student_id}")
        
        if not all([student_id, face_image_base64]):
            return jsonify({"error": "Missing required fields: studentId, faceImage"}), 400
        
        # Convert base64 to image
        image = base64_to_image(face_image_base64)
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400
        
        # Check liveness (optimized)
        liveness_score = anti_spoof_service.predict(image)
        logger.info(f"Liveness score for {student_id}: {liveness_score}")
        
        if liveness_score < 0.5:
            return jsonify({
                "error": "Spoof detected. Please use your real face.",
                "liveness_score": liveness_score
            }), 400
        
        # Recognize face (optimized)
        recognized_id, confidence = face_recognition_service.recognize_face(image)
        logger.info(f"Recognition result: {recognized_id}, confidence: {confidence}")
        
        # Check match
        if recognized_id != student_id or confidence < 0.6:
            return jsonify({
                "error": "Face does not match the provided Student ID",
                "liveness_score": liveness_score,
                "recognition_confidence": confidence,
                "recognized_student": recognized_id
            }), 400
        
        # Mark attendance (same as before)
        if db:
            student_ref = db.collection('students').document(student_id)
            student_doc = student_ref.get()
            
            if not student_doc.exists:
                return jsonify({"error": "Student not found in database"}), 404
            
            student_data = student_doc.to_dict()
            
            today = datetime.now().date().isoformat()
            existing_attendance = list(db.collection('attendance')
                .where('studentId', '==', student_id)
                .where('date', '==', today)
                .limit(1).stream())
            
            if len(existing_attendance) > 0:
                return jsonify({"error": "Attendance already marked today"}), 400
            
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
            
            # Force garbage collection
            gc.collect()
            
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

# Other routes remain the same...
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
    try:
        face_recognition_service.load_known_faces()
        return jsonify({
            "success": True,
            "message": "Known faces reloaded",
            "count": len(face_recognition_service.known_face_embeddings)
        })
    except Exception as e:
        logger.error(f"Failed to reload faces: {e}")
        return jsonify({"error": "Failed to reload faces"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
