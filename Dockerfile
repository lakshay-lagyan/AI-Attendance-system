FROM python:3.11-slim-bookworm

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
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies in specific order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir tensorflow==2.15.0 && \
    pip install --no-cache-dir keras==2.15.0 && \
    pip install --no-cache-dir tf-keras==2.15.0 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for FAISS index and data
RUN mkdir -p /app/data

# Make start script executable
RUN chmod +x /app/start.sh

# Expose port (Railway will set PORT env var at runtime)
EXPOSE 10000

# Set environment variables - FIX KERAS IMPORT ISSUE
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV TF_USE_LEGACY_KERAS=1

# Use startup script to handle PORT variable properly
CMD ["/app/start.sh"]