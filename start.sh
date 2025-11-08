#!/bin/bash

# Get PORT from environment or use default
PORT=${PORT:-10000}

echo "Starting application on port $PORT..."

# Start gunicorn
exec gunicorn main:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --threads 4 \
    --worker-class sync \
    --timeout 300 \
    --keep-alive 5 \
    --max-requests 1000 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --capture-output
