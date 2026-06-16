# =============================================================================
# Dockerfile — FastAPI Gateway (Render Free Tier)
# =============================================================================
# Deployed on: Render Web Service (Free Tier)
# =============================================================================

FROM python:3.11-slim-bookworm

# ---------------------------- Metadata ---------------------------------------
LABEL maintainer="ML Pipeline Team"
LABEL description="FastAPI inference gateway — job submission & status polling"

# ---------------------------- Environment ------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# ---------------------------- Working directory ------------------------------
WORKDIR /app

# ---------------------------- Python deps ------------------------------------
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------- App code ---------------------------------------
COPY app.py .

# ---------------------------- Health check -----------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health')" || exit 1

# ---------------------------- Entry point ------------------------------------
# Render provides $PORT env; default to 8000 for local dev
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info