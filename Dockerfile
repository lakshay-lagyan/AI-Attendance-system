# Use Python 3.11 (stable and compatible with all packages)
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgfortran5 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for FAISS index
RUN mkdir -p /tmp

# Expose port
EXPOSE 10000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=10000
ENV TF_USE_LEGACY_KERAS=0

# Run gunicorn
CMD gunicorn main:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 300 --access-logfile - --error-logfile -