"""
FIXED VERSION OF MAIN.PY - Production Ready Smart Attendance System
This file fixes all critical bugs and integrates the new super admin module
"""

from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash, session
from pymongo import MongoClient
from deepface import DeepFace
import numpy as np
import mediapipe as mp
import cv2 as cv
import pickle
import datetime
import os
import faiss
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import re
from email_validator import validate_email, EmailNotValidError
import asyncio
from concurrent.futures import ThreadPoolExecutor
from bson import ObjectId
import threading

# Import function modules
from function import generate_camera_stream, get_face_embedding, continuous_learning_update, create_3d_template, extract_multi_vector_embeddings, ensemble_matching, search_face_faiss, rebuild_faiss_index

# Import the new super admin module
from superadmin_module import register_superadmin_module

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

# Rate Limiter Configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
    strategy="fixed-window"
)

# MONGODB
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

# Email Verification System 
class EmailVerificationSystem:
    
    @staticmethod
    def verify_email_format(email: str) -> dict:
        """Verify email format and structure"""
        try:
            # Basic format check
            if '@' not in email:
                return {
                    "valid": False,
                    "reason": "Invalid email format - missing @",
                    "tag": "INVALID_FORMAT"
                }
            
            email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_regex, email):
                return {
                    "valid": False,
                    "reason": "Invalid email format",
                    "tag": "INVALID_FORMAT"
                }
            
            # Advanced validation with email-validator
            try:
                validation = validate_email(email, check_deliverability=True)
                email_normalized = validation.normalized
                
                return {
                    "valid": True,
                    "normalized": email_normalized,
                    "domain": email.split('@')[1],
                    "tag": "VERIFIED",
                    "mx_records_exist": True
                }
            except EmailNotValidError as e:
                return {
                    "valid": False,
                    "reason": str(e),
                    "tag": "INVALID_DOMAIN"
                }
                
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Verification error: {str(e)}",
                "tag": "ERROR"
            }
    
    @staticmethod
    def is_disposable_email(email: str) -> bool:
        """Check if email is from disposable domain"""
        try:
            disposable_domains = [
                'tempmail.com', 'guerrillamail.com', '10minutemail.com',
                'mailinator.com', 'throwaway.email', 'temp-mail.org'
            ]
            domain = email.split('@')[1].lower()
            return domain in disposable_domains
        except:
            return False

email_verifier = EmailVerificationSystem()

# FAISS Vector Database Setup - FIXED
EMBEDDING_DIM = 512
faiss_index = None
person_id_map = []

def initialize_faiss_index():
    """Initialize FAISS index properly"""
    global faiss_index, person_id_map
    index_path = "faiss_index.bin"
    map_path = "person_id_map.pkl"
    
    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(map_path, 'rb') as f:
                person_id_map = pickle.load(f)
            print(f"Loaded FAISS index with {faiss_index.ntotal} faces")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            person_id_map = []
    else:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        person_id_map = []
        print("Created new FAISS index")

def save_faiss_index():
    """Save FAISS index to disk"""
    try:
        if faiss_index is not None:
            faiss.write_index(faiss_index, "faiss_index.bin")
            with open("person_id_map.pkl", 'wb') as f:
                pickle.dump(person_id_map, f)
    except Exception as e:
        print(f"Error saving FAISS index: {e}")

# Initialize FAISS on startup
initialize_faiss_index()

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class SuperAdminUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc.get("_id"))
        self.email = doc.get("email")
        self.name = doc.get("name", "Super Admin")
        self.role = "superadmin"
        self.profile_image = doc.get("profile_image", "")

class AdminUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc.get("_id"))
        self.email = doc.get("email")
        self.name = doc.get("name", "Admin")
        self.role = "admin"
        self.profile_image = doc.get("profile_image", "")

class RegularUser(UserMixin):
    def __init__(self, doc):
        self.doc = doc
        self.id = str(doc.get("_id"))
        self.email = doc.get("email")
        self.name = doc.get("name", "User")
        self.role = "user"
        self.department = doc.get("department", "")
        self.profile_image = doc.get("profile_image", "")
        self.status = doc.get("status", "active")

@login_manager.user_loader
def load_user(user_id):
    # Try super admin first
    doc = superadmins_col.find_one({"_id": user_id})
    if doc:
        return SuperAdminUser(doc)
    
    # Try admin
    doc = admins_col.find_one({"_id": user_id})
    if doc:
        return AdminUser(doc)
    
    # Try regular user
    doc = users_col.find_one({"_id": user_id})
    if doc:
        return RegularUser(doc)
    
    # Fallback with ObjectId
    try:
        doc = superadmins_col.find_one({"_id": ObjectId(user_id)})
        if doc:
            return SuperAdminUser(doc)
        doc = admins_col.find_one({"_id": ObjectId(user_id)})
        if doc:
            return AdminUser(doc)
        doc = users_col.find_one({"_id": ObjectId(user_id)})
        if doc:
            return RegularUser(doc)
    except:
        pass
    return None

def superadmin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'superadmin':
            flash('Access denied. Super Admin privileges required.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
                                                                                                                                                                                            
def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role not in ['admin', 'superadmin']:
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# Register the super admin module
register_superadmin_module(app)

# --- Authentication Routes ---
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == 'superadmin':
            return redirect(url_for("superadmin.dashboard"))
        elif current_user.role == 'admin':
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    try:
        if current_user.is_authenticated:
            if current_user.role == 'superadmin':
                return redirect(url_for("superadmin.dashboard"))
            elif current_user.role == 'admin':
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("user_dashboard"))
        
        if request.method == "GET":
            try:
                return render_template("login_simple.html")
            except Exception as template_error:
                print(f"❌ Template rendering error: {template_error}")
                # Fallback to basic HTML if template fails
                return '''
                <html><body style="font-family:sans-serif;max-width:400px;margin:100px auto;padding:20px;">
                <h2>Sign In</h2>
                <form method="POST">
                <input name="email" type="email" placeholder="Email" required style="width:100%;padding:10px;margin:10px 0;"><br>
                <input name="password" type="password" placeholder="Password" required style="width:100%;padding:10px;margin:10px 0;"><br>
                <button type="submit" style="width:100%;padding:10px;background:#3b82f6;color:white;border:none;cursor:pointer;">Sign In</button>
                </form>
                </body></html>
                ''', 200
        
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        remember = request.form.get("remember") == "on"
        
        if not email or not password:
            flash("Email and password required.", "danger")
            return redirect(url_for("login"))
        
        # Check super admin first
        try:
            superadmin = superadmins_col.find_one({"email": email})
            if superadmin:
                hashed = superadmin.get("password_hash")
                if hashed and check_password_hash(hashed, password):
                    user = SuperAdminUser(superadmin)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    if next_page and next_page != '/logout':
                        return redirect(next_page)
                    return redirect(url_for("superadmin.dashboard"))
        except Exception as e:
            print(f"Superadmin login error: {e}")
        
        # Check admin
        try:
            admin = admins_col.find_one({"email": email})
            if admin:
                hashed = admin.get("password_hash")
                if hashed and check_password_hash(hashed, password):
                    user = AdminUser(admin)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    if next_page and next_page != '/logout':
                        return redirect(next_page)
                    return redirect(url_for("admin_dashboard"))
        except Exception as e:
            print(f"Admin login error: {e}")
        
        # Check regular user
        try:
            user_doc = users_col.find_one({"email": email})
            if user_doc:
                if user_doc.get("status") == "blocked":
                    flash("Your account has been blocked. Contact admin.", "danger")
                    return redirect(url_for("login"))
                
                hashed = user_doc.get("password_hash")
                if hashed and check_password_hash(hashed, password):
                    user = RegularUser(user_doc)
                    login_user(user, remember=remember)
                    next_page = request.args.get('next')
                    if next_page and next_page != '/logout':
                        return redirect(next_page)
                    return redirect(url_for("user_dashboard"))
        except Exception as e:
            print(f"User login error: {e}")
        
        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))
    
    except Exception as e:
        print(f"❌ Login route error: {e}")
        return '''
        <html><body style="font-family:sans-serif;max-width:400px;margin:100px auto;padding:20px;background:#0f172a;color:#fff;">
        <h2 style="color:#ef4444;">System Error</h2>
        <p>Please try again later or contact support.</p>
        <a href="/login" style="color:#3b82f6;">Back to Login</a>
        </body></html>
        ''', 500

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# --- Admin Routes ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin Dashboard"""
    try:
        total_users = persons_col.count_documents({})
        total_attendance = attendance_col.count_documents({})
        pending_requests = enrollment_requests_col.count_documents({"status": "pending"})
        today_attendance = attendance_col.count_documents({
            "timestamp": {
                "$gte": datetime.datetime.combine(datetime.date.today(), datetime.time.min),
                "$lt": datetime.datetime.combine(datetime.date.today(), datetime.time.max)
            }
        })
        
        return render_template("admin_dashboard.html",
                             admin=current_user,
                             total_users=total_users,
                             total_attendance=total_attendance,
                             pending_requests=pending_requests,
                             today_attendance=today_attendance)
    except Exception as e:
        print(f"Admin dashboard error: {e}")
        flash("Error loading dashboard. Please try again.", "danger")
        return render_template("admin_dashboard.html",
                             admin=current_user,
                             total_users=0,
                             total_attendance=0,
                             pending_requests=0,
                             today_attendance=0)

# --- User Routes ---
@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if current_user.role == 'admin':
        return redirect(url_for("admin_dashboard"))
    if current_user.role == 'superadmin':
        return redirect(url_for("superadmin.dashboard"))
    return render_template("user_dashboard_new.html", user=current_user)

# --- Mobile Attendance Routes (FIXED) ---
@app.route("/api/mobile/mark_attendance", methods=["POST"])
@limiter.limit("20 per hour")
def mobile_mark_attendance():
    """Mark attendance from mobile camera capture - FIXED VERSION"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "failed", "msg": "No image provided"}), 400
        
        # Get face embedding
        embedding = get_face_embedding(file)
        if embedding is None:
            return jsonify({"status": "failed", "msg": "No face detected. Please ensure your face is clearly visible and try again."}), 400
        
        # Search in FAISS - FIXED: Use correct variable names
        if faiss_index is None or faiss_index.ntotal == 0:
            return jsonify({"status": "failed", "msg": "System not initialized. Please contact administrator."}), 500
        
        # Use the correct search function
        matched_name, confidence = search_face_faiss(embedding, threshold=0.6)
        
        if not matched_name:
            return jsonify({
                "status": "failed",
                "msg": "Face not recognized. Please enroll first or contact administrator."
            }), 404
        
        # Get person details
        person = persons_col.find_one({"name": matched_name})
        if not person:
            return jsonify({"status": "failed", "msg": "Person data not found"}), 404
        
        # Check if already marked today
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        existing = attendance_col.find_one({
            "name": matched_name,
            "timestamp": {"$gte": today_start}
        })
        
        if existing:
            return jsonify({
                "status": "success",
                "message": "Attendance already marked for today",
                "person_name": person.get("name", "Unknown"),
                "time": existing["timestamp"].strftime("%I:%M %p") if isinstance(existing["timestamp"], datetime.datetime) else "Unknown",
                "confidence": round(existing.get("confidence", 0) * 100, 2)
            })
        
        # Mark attendance
        attendance_col.insert_one({
            "name": matched_name,
            "timestamp": datetime.datetime.utcnow(),
            "confidence": confidence,
            "device": "mobile",
            "user_agent": request.headers.get("User-Agent", ""),
            "location": request.headers.get("X-Forwarded-For", request.remote_addr)
        })
        
        # Log attendance event
        system_logs_col.insert_one({
            "action": "mobile_attendance",
            "person_name": matched_name,
            "confidence": confidence,
            "device": "mobile",
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({
            "status": "success",
            "message": "Attendance marked successfully!",
            "person_name": person.get("name", "Unknown"),
            "time": datetime.datetime.now().strftime("%I:%M %p"),
            "confidence": round(confidence * 100, 2)
        })
        
    except Exception as e:
        print(f"[Mobile Attendance Error] {e}")
        return jsonify({"status": "failed", "msg": "Server error. Please try again or contact administrator."}), 500

# --- API Routes for Stats (FIXED) ---
@app.route("/api/superadmin/stats")
@superadmin_required
def superadmin_stats_api():
    """API endpoint for super admin statistics - FIXED"""
    try:
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        today_end = datetime.datetime.combine(datetime.date.today(), datetime.time.max)
        
        stats = {
            "total_users": users_col.count_documents({}),
            "total_admins": admins_col.count_documents({}),
            "total_persons": persons_col.count_documents({"status": {"$ne": "blocked"}}),
            "pending_enrollments": enrollment_requests_col.count_documents({"status": "pending"}),
            "today_attendance": attendance_col.count_documents({
                "timestamp": {
                    "$gte": today_start,
                    "$lt": today_end
                }
            }),
            "total_cameras": cameras_col.count_documents({}),
            "active_cameras": 0,  # Will be updated by camera module
            "faiss_index_size": faiss_index.ntotal if faiss_index else 0,
            "system_health": "healthy"
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Video Feed Route ---
@app.route("/video_feed")
@admin_required
def video_feed():
    return Response(generate_camera_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Create Super Admin Route ---
@app.route("/create_superadmin", methods=["POST"])
def create_superadmin():
    """Create super admin account"""
    secret = os.environ.get("CREATE_SUPERADMIN_SECRET", "temp_secret_123")
    provided = request.form.get("secret", "")
    
    if provided != secret:
        return jsonify({"status": "failed", "msg": "Invalid secret"}), 403
    
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    
    if not (name and email and password):
        return jsonify({"status": "failed", "msg": "Missing fields"}), 400
    
    email = email.strip().lower()
    existing = superadmins_col.find_one({"email": email})
    if existing:
        return jsonify({"status": "failed", "msg": "Super admin already exists"}), 400
    
    password_hash = generate_password_hash(password)
    doc = {
        "_id": str(datetime.datetime.utcnow().timestamp()) + "_" + email,
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "profile_image": "",
        "created_at": datetime.datetime.utcnow()
    }
    superadmins_col.insert_one(doc)
    return jsonify({"status": "success", "msg": "Super admin created"}), 201

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
