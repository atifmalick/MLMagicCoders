# =============================================================================
# Dockerfile — Hugging Face Spaces Inference Worker
# =============================================================================
# Deployed on: Hugging Face Spaces (Free CPU Basic Instance)
# Base image: official PyTorch with CUDA runtime stripped for smallest CPU size
# =============================================================================

FROM python:3.11-slim-bookworm

# ---------------------------- Metadata ---------------------------------------
LABEL maintainer="ML Pipeline Team"
LABEL description="Inference worker that polls Supabase and runs distilbert"

# ---------------------------- Build-time args --------------------------------
ARG WORKDIR=/app

# ---------------------------- Environment ------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ---------------------------- System deps ------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------- Working directory ------------------------------
WORKDIR ${WORKDIR}

# ---------------------------- Python deps ------------------------------------
COPY requirements-worker.txt .

RUN pip install --no-cache-dir -r requirements-worker.txt

# ---------------------------- App code ---------------------------------------
COPY worker.py .

# ---------------------------- Entry point ------------------------------------
CMD ["python", "-u", "worker.py"]