FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch CPU version
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Clone Silent Face Anti-Spoofing
RUN git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git /app/silent-antispoofing

# Download anti-spoofing models
RUN mkdir -p /app/silent-antispoofing/resources/anti_spoof_models && \
    cd /app/silent-antispoofing/resources/anti_spoof_models && \
    wget -O 2.7_80x80_MiniFASNetV2.pth https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases/download/v2.0/2.7_80x80_MiniFASNetV2.pth && \
    wget -O 4_0_0_80x80_MiniFASNetV1SE.pth https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/releases/download/v2.0/4_0_0_80x80_MiniFASNetV1SE.pth

# Copy application code
COPY app.py .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]