#!/bin/bash

# Production startup script for Smart Attendance System
# This script properly initializes the system and starts the server

echo "🚀 Starting Smart Attendance System..."

# Set environment variables for TensorFlow/Keras compatibility
export TF_CPP_MIN_LOG_LEVEL=3
export TF_ENABLE_ONEDNN_OPTS=0
export TF_USE_LEGACY_KERAS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

# Create necessary directories
mkdir -p logs
mkdir -p data

# Initialize FAISS index if it doesn't exist
if [ ! -f "faiss_index.bin" ]; then
    echo "🔧 Initializing FAISS index..."
    python3 -c "
import faiss
import pickle
import numpy as np

# Create empty FAISS index
index = faiss.IndexFlatIP(512)
faiss.write_index(index, 'faiss_index.bin')

# Create empty person ID map
with open('person_id_map.pkl', 'wb') as f:
    pickle.dump([], f)

print('✅ FAISS index initialized')
"
fi

# Test critical imports
echo "🧪 Testing critical imports..."
python3 test_imports.py
if [ $? -ne 0 ]; then
    echo "❌ Import test failed. Please check dependencies."
    exit 1
fi

# Get port from environment or use default
PORT=${PORT:-5000}

echo "🌐 Starting server on port $PORT..."

# Start the application using the fixed main file
if [ "$ENVIRONMENT" = "production" ]; then
    # Production mode with gunicorn
    exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --keep-alive 2 main_fixed:app
else
    # Development mode
    exec python3 main_fixed.py
fi
