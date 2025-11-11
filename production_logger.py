"""
Production-Grade Logging System
Tracks all errors, performance metrics, and system events
"""
import logging
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
import datetime
import traceback
from functools import wraps
import time

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Configure logging
def setup_production_logging():
    """Setup comprehensive logging for production"""
    
    # Main application logger
    app_logger = logging.getLogger('app')
    app_logger.setLevel(logging.INFO)
    
    # Error logger
    error_logger = logging.getLogger('error')
    error_logger.setLevel(logging.ERROR)
    
    # Performance logger
    perf_logger = logging.getLogger('performance')
    perf_logger.setLevel(logging.INFO)
    
    # Face recognition logger
    face_logger = logging.getLogger('face_recognition')
    face_logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # JSON formatter for structured logging
    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_obj = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno
            }
            if record.exc_info:
                log_obj['exception'] = traceback.format_exception(*record.exc_info)
            return json.dumps(log_obj)
    
    json_formatter = JSONFormatter()
    
    # File handler - rotating by size (10MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    app_logger.addHandler(file_handler)
    
    # Error file handler
    error_handler = RotatingFileHandler(
        'logs/error.log',
        maxBytes=10*1024*1024,
        backupCount=10
    )
    error_handler.setFormatter(formatter)
    error_logger.addHandler(error_handler)
    
    # Performance file handler - JSON format
    perf_handler = TimedRotatingFileHandler(
        'logs/performance.json',
        when='midnight',
        interval=1,
        backupCount=30
    )
    perf_handler.setFormatter(json_formatter)
    perf_logger.addHandler(perf_handler)
    
    # Face recognition detailed logs
    face_handler = RotatingFileHandler(
        'logs/face_recognition.log',
        maxBytes=20*1024*1024,
        backupCount=5
    )
    face_handler.setFormatter(formatter)
    face_logger.addHandler(face_handler)
    
    # Console handler for development
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app_logger.addHandler(console_handler)
    
    return app_logger, error_logger, perf_logger, face_logger

# Initialize loggers
app_logger, error_logger, perf_logger, face_logger = setup_production_logging()

def log_error(func):
    """Decorator to log errors in functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_logger.error(
                f"Error in {func.__name__}: {str(e)}",
                exc_info=True,
                extra={
                    'function': func.__name__,
                    'args': str(args)[:200],
                    'kwargs': str(kwargs)[:200]
                }
            )
            raise
    return wrapper

def log_performance(func):
    """Decorator to log performance metrics"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        perf_logger.info(json.dumps({
            'function': func.__name__,
            'execution_time': execution_time,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }))
        
        # Warning for slow operations
        if execution_time > 1.0:
            app_logger.warning(
                f"Slow operation: {func.__name__} took {execution_time:.2f}s"
            )
        
        return result
    return wrapper

def log_face_recognition(event_type, details):
    """Log face recognition events"""
    face_logger.info(json.dumps({
        'event_type': event_type,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        **details
    }))

class SystemHealthMonitor:
    """Monitor system health metrics"""
    
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'database_errors': 0,
            'api_errors': 0,
            'average_response_time': 0.0
        }
        self.response_times = []
    
    def record_request(self, response_time, success=True):
        self.metrics['total_requests'] += 1
        self.response_times.append(response_time)
        
        # Keep last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        self.metrics['average_response_time'] = sum(self.response_times) / len(self.response_times)
        
        if success:
            self.metrics['successful_recognitions'] += 1
        else:
            self.metrics['failed_recognitions'] += 1
    
    def record_error(self, error_type):
        if error_type == 'database':
            self.metrics['database_errors'] += 1
        elif error_type == 'api':
            self.metrics['api_errors'] += 1
    
    def get_health_status(self):
        total = self.metrics['total_requests']
        if total == 0:
            return {'status': 'healthy', 'metrics': self.metrics}
        
        success_rate = (self.metrics['successful_recognitions'] / total) * 100
        
        if success_rate >= 95:
            status = 'healthy'
        elif success_rate >= 80:
            status = 'degraded'
        else:
            status = 'unhealthy'
        
        return {
            'status': status,
            'success_rate': round(success_rate, 2),
            'metrics': self.metrics
        }
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            with open('logs/system_metrics.json', 'w') as f:
                json.dump({
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    **self.get_health_status()
                }, f, indent=2)
        except Exception as e:
            error_logger.error(f"Failed to save metrics: {e}")

# Global health monitor
health_monitor = SystemHealthMonitor()

# Export
__all__ = [
    'app_logger',
    'error_logger',
    'perf_logger',
    'face_logger',
    'log_error',
    'log_performance',
    'log_face_recognition',
    'health_monitor'
]
