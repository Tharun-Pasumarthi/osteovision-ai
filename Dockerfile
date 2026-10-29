# =====================================================================
# Production Python/FastAPI Backend for Osteoporosis Screening
# Optimized for CPU Cloud Hosting (Render, Railway, Hugging Face)
# =====================================================================

FROM python:3.12-slim

# Install essential system dependencies (libglib, libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch (~180MB instead of 2.8GB CUDA bundle)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy backend application code, ML pipelines, and configurations
COPY backend ./backend
COPY src ./src
COPY configs ./configs
COPY outputs ./outputs
COPY frontend/dist ./frontend/dist
RUN mkdir -p /app/data/metadata

# Production environment variables: single-thread OpenMP to respect 512MB RAM
ENV HOST=0.0.0.0 \
    PORT=10000 \
    PYTHONUNBUFFERED=1 \
    USE_CUDA=0 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1

EXPOSE 10000 7860

# Start FastAPI server
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
