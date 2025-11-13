"""
Production-Grade Super Admin Module for Smart Attendance System
Fixes all identified issues and provides comprehensive admin functionality
"""

from flask import Blueprint, render_template, request, jsonify, Response, redirect, url_for, flash, current_app
from pymongo import MongoClient
import datetime
import os
import cv2 as cv
import threading
import numpy as np
from bson import ObjectId
from werkzeug.security import generate_password_hash
from flask_login import current_user, login_required
from functools import wraps
import traceback

# Create Blueprint
superadmin_bp = Blueprint('superadmin', __name__, url_prefix='/superadmin')

# Global variables for camera management
active_camera_streams = {}
camera_stream_locks = {}

def superadmin_required(f):
    """Decorator to require super admin privileges"""
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if not hasattr(current_user, 'role') or current_user.role != 'superadmin':
            flash('Access denied. Super Admin privileges required.', 'danger')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def get_db_collections():
    """Get database collections"""
    MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10000)
    
    transactional_db = client["transactional_db"]
    attendance_col = transactional_db["attendance"]
    
    core = client['secure_db']
    persons_col = core["persons"]
    profile_col = core["profile"]
    superadmins_col = core["superadmins"]
    admins_col = core["admins"]
    users_col = core["users"]
    enrollment_requests_col = core["enrollment_requests"]
    system_logs_col = core["system_logs"]
    cameras_col = core["cameras"]
    
    return {
        'attendance_col': attendance_col,
        'persons_col': persons_col,
        'profile_col': profile_col,
        'superadmins_col': superadmins_col,
        'admins_col': admins_col,
        'users_col': users_col,
        'enrollment_requests_col': enrollment_requests_col,
        'system_logs_col': system_logs_col,
        'cameras_col': cameras_col
    }

# Dashboard Routes
@superadmin_bp.route('/dashboard')
@superadmin_required
def dashboard():
    """Super Admin Dashboard"""
    try:
        db = get_db_collections()
        
        stats = {
            'total_admins': db['admins_col'].count_documents({}),
            'total_users': db['persons_col'].count_documents({}),
            'total_attendance': db['attendance_col'].count_documents({}),
            'pending_requests': db['enrollment_requests_col'].count_documents({"status": "pending"}),
            'active_cameras': len(active_camera_streams),
            'total_cameras': db['cameras_col'].count_documents({})
        }
        
        return render_template('superadmin/dashboard_new.html', 
                             superadmin=current_user, 
                             stats=stats)
    except Exception as e:
        flash(f'Error loading dashboard: {str(e)}', 'danger')
        return render_template('superadmin/dashboard_new.html', 
                             superadmin=current_user, 
                             stats={})

# API Routes
@superadmin_bp.route('/api/stats')
@superadmin_required
def api_stats():
    """Get comprehensive statistics"""
    try:
        db = get_db_collections()
        
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        today_end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
        
        stats = {
            "total_superadmins": db['superadmins_col'].count_documents({}),
            "total_admins": db['admins_col'].count_documents({}),
            "total_users": db['users_col'].count_documents({}),
            "total_persons": db['persons_col'].count_documents({"status": {"$ne": "blocked"}}),
            "active_users": db['users_col'].count_documents({"status": "active"}),
            "blocked_users": db['users_col'].count_documents({"status": "blocked"}),
            "pending_enrollments": db['enrollment_requests_col'].count_documents({"status": "pending"}),
            "today_attendance": db['attendance_col'].count_documents({
                "timestamp": {"$gte": today_start, "$lt": today_end}
            }),
            "total_attendance": db['attendance_col'].count_documents({}),
            "total_cameras": db['cameras_col'].count_documents({}),
            "active_cameras": len(active_camera_streams),
            "system_health": "healthy"
        }
        
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Camera Management
@superadmin_bp.route('/cameras')
@superadmin_required
def cameras():
    """Camera management page"""
    return render_template('superadmin/cameras_new.html', superadmin=current_user)

@superadmin_bp.route('/api/cameras', methods=['GET'])
@superadmin_required
def api_get_cameras():
    """Get all cameras"""
    try:
        db = get_db_collections()
        cameras = list(db['cameras_col'].find({}))
        
        for cam in cameras:
            cam["_id"] = str(cam["_id"])
            cam["is_active"] = str(cam["_id"]) in active_camera_streams
            
            if "created_at" in cam and cam["created_at"]:
                try:
                    cam["created_at"] = cam["created_at"].isoformat()
                except:
                    pass
        
        return jsonify({"status": "success", "cameras": cameras}), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras', methods=['POST'])
@superadmin_required
def api_create_camera():
    """Create new camera"""
    try:
        db = get_db_collections()
        data = request.get_json()
        
        name = data.get("name", "").strip()
        source_type = data.get("source_type", "opencv")
        
        if not name:
            return jsonify({"status": "failed", "error": "Camera name is required"}), 400
        
        # Check for duplicate name
        if db['cameras_col'].find_one({"name": name}):
            return jsonify({"status": "failed", "error": "Camera name already exists"}), 400
        
        # Determine source
        if source_type == "opencv":
            camera_index = data.get("camera_index")
            if camera_index is None:
                return jsonify({"status": "failed", "error": "Camera index is required"}), 400
            source = int(camera_index)
        else:
            stream_url = data.get("stream_url", "").strip()
            if not stream_url:
                return jsonify({"status": "failed", "error": "Stream URL is required"}), 400
            source = stream_url
        
        camera_id = f"cam_{int(datetime.datetime.utcnow().timestamp())}_{name.replace(' ', '_')[:20]}"
        
        camera_doc = {
            "_id": camera_id,
            "name": name,
            "source_type": source_type,
            "source": source,
            "config": {
                "fps": data.get("fps", 30),
                "resolution": {
                    "width": data.get("resolution_width", 640),
                    "height": data.get("resolution_height", 480)
                },
                "enabled": True
            },
            "enabled": True,
            "is_active": False,
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email,
            "last_seen": None
        }
        
        db['cameras_col'].insert_one(camera_doc)
        
        # Log action
        db['system_logs_col'].insert_one({
            "action": "create_camera",
            "camera_id": camera_id,
            "camera_name": name,
            "performed_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": f"Camera '{name}' created successfully",
            "camera_id": camera_id
        }), 201
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>/start', methods=['POST'])
@superadmin_required
def api_start_camera(camera_id):
    """Start camera stream"""
    try:
        db = get_db_collections()
        camera = db['cameras_col'].find_one({"_id": camera_id})
        
        if not camera:
            return jsonify({"status": "failed", "error": "Camera not found"}), 404
        
        if camera_id in active_camera_streams:
            return jsonify({"status": "success", "message": "Camera already active"}), 200
        
        # Start camera in background thread
        def start_camera_thread():
            try:
                source = camera["source"]
                cap = cv.VideoCapture(source)
                
                if not cap.isOpened():
                    print(f"[Camera {camera_id}] Failed to open source: {source}")
                    return
                
                # Set resolution
                resolution = camera.get("config", {}).get("resolution", {})
                if resolution.get("width"):
                    cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution["width"])
                if resolution.get("height"):
                    cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution["height"])
                
                # Store capture object
                active_camera_streams[camera_id] = {
                    "capture": cap,
                    "started_at": datetime.datetime.utcnow()
                }
                
                # Update database
                db['cameras_col'].update_one(
                    {"_id": camera_id},
                    {"$set": {"last_seen": datetime.datetime.utcnow(), "is_active": True}}
                )
                
                print(f"[Camera {camera_id}] Started successfully")
                
            except Exception as e:
                print(f"[Camera {camera_id}] Start error: {e}")
        
        thread = threading.Thread(target=start_camera_thread, daemon=True)
        thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Camera started successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>/stop', methods=['POST'])
@superadmin_required
def api_stop_camera(camera_id):
    """Stop camera stream"""
    try:
        db = get_db_collections()
        
        if camera_id not in active_camera_streams:
            return jsonify({"status": "success", "message": "Camera already stopped"}), 200
        
        # Release camera
        stream_data = active_camera_streams[camera_id]
        if "capture" in stream_data:
            stream_data["capture"].release()
        
        del active_camera_streams[camera_id]
        
        # Update database
        db['cameras_col'].update_one(
            {"_id": camera_id},
            {"$set": {"is_active": False}}
        )
        
        return jsonify({
            "status": "success",
            "message": "Camera stopped successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/cameras/<camera_id>', methods=['DELETE'])
@superadmin_required
def api_delete_camera(camera_id):
    """Delete camera"""
    try:
        db = get_db_collections()
        
        # Stop camera if active
        if camera_id in active_camera_streams:
            stream_data = active_camera_streams[camera_id]
            if "capture" in stream_data:
                stream_data["capture"].release()
            del active_camera_streams[camera_id]
        
        # Delete from database
        result = db['cameras_col'].delete_one({"_id": camera_id})
        
        if result.deleted_count == 0:
            return jsonify({"status": "failed", "error": "Camera not found"}), 404
        
        return jsonify({
            "status": "success",
            "message": "Camera deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

# Admin Management
@superadmin_bp.route('/admins')
@superadmin_required
def admins():
    """Admin management page"""
    return render_template('superadmin/admins_new.html', superadmin=current_user)

@superadmin_bp.route('/api/admins', methods=['GET'])
@superadmin_required
def api_get_admins():
    """Get all admins"""
    try:
        db = get_db_collections()
        admins = list(db['admins_col'].find({}, {"password_hash": 0}))
        
        for admin in admins:
            admin["_id"] = str(admin["_id"])
            admin["id"] = admin["_id"]
            
            if "created_at" in admin and admin["created_at"]:
                try:
                    admin["created_at"] = admin["created_at"].isoformat()
                except:
                    admin["created_at"] = None
            
            # Ensure required fields
            admin["name"] = admin.get("name", "Unknown")
            admin["email"] = admin.get("email", "")
            admin["department"] = admin.get("department", "")
            admin["is_active"] = admin.get("is_active", True)
        
        return jsonify({"status": "success", "admins": admins}), 200
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@superadmin_bp.route('/api/create_admin', methods=['POST'])
@superadmin_required
def api_create_admin():
    """Create new admin"""
    try:
        db = get_db_collections()
        data = request.get_json()
        
        name = data.get("name", "").strip()
        email = data.get("email", "").strip().lower()
        password = data.get("password", "")
        department = data.get("department", "").strip()
        
        if not all([name, email, password]):
            return jsonify({
                "status": "failed",
                "error": "Name, email and password are required"
            }), 400
        
        # Check if admin exists
        if db['admins_col'].find_one({"email": email}):
            return jsonify({
                "status": "failed",
                "error": "Admin with this email already exists"
            }), 400
        
        # Create admin
        admin_id = f"{datetime.datetime.utcnow().timestamp()}_{email}"
        admin_doc = {
            "_id": admin_id,
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "department": department if department else None,
            "profile_image": "",
            "is_active": True,
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email
        }
        
        db['admins_col'].insert_one(admin_doc)
        
        # Log action
        db['system_logs_col'].insert_one({
            "action": "create_admin",
            "admin_email": email,
            "performed_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": f"Admin '{name}' created successfully"
        }), 201
        
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

# Register blueprint
def register_superadmin_module(app):
    """Register the super admin module with the Flask app"""
    app.register_blueprint(superadmin_bp)
    return superadmin_bp
