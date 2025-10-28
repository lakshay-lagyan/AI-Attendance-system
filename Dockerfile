FROM python:3.11-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for FAISS
RUN mkdir -p /app/data

# Expose port
EXPOSE 10000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
ENV TF_USE_LEGACY_KERAS=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:10000/health')" || exit 1

# Run with gunicorn
CMD gunicorn main:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 2 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    --log-level info