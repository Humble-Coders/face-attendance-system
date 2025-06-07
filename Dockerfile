FROM python:3.9-slim

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

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git /app/silent-antispoofing

# Download models from working URLs
RUN mkdir -p /app/silent-antispoofing/resources/anti_spoof_models && \
    cd /app/silent-antispoofing/resources/anti_spoof_models && \
    wget -O 2.7_80x80_MiniFASNetV2.pth https://huggingface.co/datasets/datasets-examples/file-examples-from-koding/resolve/main/FAS_Models/2.7_80x80_MiniFASNetV2.pth && \
    wget -O 4_0_0_80x80_MiniFASNetV1SE.pth https://huggingface.co/datasets/datasets-examples/file-examples-from-koding/resolve/main/FAS_Models/4_0_0_80x80_MiniFASNetV1SE.pth

COPY app.py .

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]
