FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    YOLO_CONFIG_DIR=/app/yolo_config \
    INSIGHTFACE_HOME=/app/insightface_models

# ffmpeg + opencv runtime libs + cmake/g++ for dlib build
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl \
        cmake g++ build-essential libopenblas-dev \
        tzdata && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO models so first start is fast.
# YOLO26s = ~25 MB (state-of-the-art for 2026, NMS-free).
# YOLOv8s = ~22 MB fallback.
RUN mkdir -p /app/yolo_config && \
    python -c "from ultralytics import YOLO; YOLO('yolo26s.pt')" && \
    python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')" && \
    echo "YOLO models downloaded"

# Pre-download InsightFace buffalo_l (~280 MB) — SCRFD detection + ArcFace.
RUN mkdir -p /app/insightface_models && \
    python -c "import insightface; \
app = insightface.app.FaceAnalysis(name='buffalo_l', root='/app/insightface_models', \
    providers=['CPUExecutionProvider'], allowed_modules=['detection','recognition']); \
app.prepare(ctx_id=-1, det_size=(640,640))" && \
    echo "InsightFace buffalo_l downloaded"

COPY app ./app
COPY static ./static

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
