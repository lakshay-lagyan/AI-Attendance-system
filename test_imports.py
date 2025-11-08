#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test script to verify all critical imports work correctly.
Run this before deploying to catch any import errors early.
"""

import sys
import os

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set environment variables that will be set in production
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

print("=" * 60)
print("Testing Critical Imports for Railway Deployment")
print("=" * 60)

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {package_name or module_name}: {e}")
        return False

# Test critical imports
results = []

print("\n📦 Core Dependencies:")
results.append(test_import("flask", "Flask"))
results.append(test_import("pymongo", "PyMongo"))
results.append(test_import("werkzeug", "Werkzeug"))
results.append(test_import("flask_login", "Flask-Login"))

print("\n🧠 Machine Learning:")
results.append(test_import("numpy", "NumPy"))
results.append(test_import("tensorflow", "TensorFlow"))

# Test Keras specifically (the problematic one)
print("\n🔧 Keras Import Test (Critical):")
try:
    import tensorflow as tf
    keras_module = tf.keras
    print(f"✅ Keras (via tf.keras) - Version: {tf.__version__}")
    results.append(True)
except Exception as e:
    print(f"❌ Keras import failed: {e}")
    results.append(False)

# Try tf_keras package
try:
    import tf_keras
    print(f"✅ tf-keras package installed")
    results.append(True)
except ImportError:
    print(f"⚠️  tf-keras package not found (may cause issues)")
    results.append(False)

print("\n👁️  Computer Vision:")
results.append(test_import("cv2", "OpenCV"))
results.append(test_import("mediapipe", "MediaPipe"))
results.append(test_import("PIL", "Pillow"))

print("\n🔍 Face Recognition:")
try:
    from deepface import DeepFace
    print(f"✅ DeepFace")
    results.append(True)
except Exception as e:
    print(f"❌ DeepFace: {e}")
    results.append(False)

print("\n🚀 Vector Search:")
results.append(test_import("faiss", "FAISS"))

print("\n🔐 Security:")
results.append(test_import("cryptography", "Cryptography"))

# Summary
print("\n" + "=" * 60)
total = len(results)
passed = sum(results)
failed = total - passed
success_rate = (passed / total) * 100

print(f"Test Results: {passed}/{total} passed ({success_rate:.1f}%)")

if failed == 0:
    print("🎉 All imports successful! Ready for deployment.")
    sys.exit(0)
else:
    print(f"⚠️  {failed} import(s) failed. Fix these before deploying.")
    sys.exit(1)
