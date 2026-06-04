# CUDA 12.1 + cuDNN8 on Ubuntu 22.04 (standard Python, no conda)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch with CUDA 12.1 — must come before openai-whisper so it picks up the GPU build
RUN python3 -m pip install --no-cache-dir \
    torch==2.3.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Pre-download model at build time → instant container startup.
# Override at build: --build-arg WHISPER_MODEL=large-v3
ARG WHISPER_MODEL=medium
ENV WHISPER_MODEL=${WHISPER_MODEL}

RUN python3 -c "import whisper; whisper.load_model('${WHISPER_MODEL}')"

COPY app/ ./app/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
