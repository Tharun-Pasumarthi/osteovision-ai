# =====================================================================
# Multi-stage Dockerfile for Osteoporosis Screening Application
# Stage 1: Build React/Vite Frontend
# Stage 2: Production Python/FastAPI Backend Serving API & Static SPA
# =====================================================================

# Stage 1: Frontend Build
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Backend & Runner
FROM python:3.12-slim AS runner

# Install system dependencies (libglib, libgl for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application, ML pipelines, and configurations
COPY backend ./backend
COPY src ./src
COPY configs ./configs
COPY outputs ./outputs
COPY data/metadata ./data/metadata

# Copy built frontend SPA from builder stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Set production environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    USE_CUDA=0

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

