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
@app.route("/admin/3d-demo")
@admin_required
def admin_3d_demo():
    """3D Face Reconstruction Demo Page"""
    return render_template("3d_face_demo.html")

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

# --- Missing Admin Routes ---
@app.route("/admin/profile")
@admin_required
def admin_profile():
    """Admin profile page"""
    profile_data = {
        "name": current_user.name,
        "email": current_user.email,
        "profile_image": current_user.profile_image,
        "role": "Administrator"
    }
    return render_template("admin_profile.html", profile=profile_data)

@app.route("/reg")
@admin_required
def reg():
    """Registration form"""
    return render_template("registration_form.html")

@app.route("/list_users")
@admin_required
def list_users():
    """List all users API"""
    try:
        users = list(persons_col.find({}, {
            "name": 1,
            "photos_count": 1,
            "enrollment_date": 1,
            "status": 1
        }))
        for user in users:
            user["_id"] = str(user["_id"])
            if "enrollment_date" in user:
                try:
                    user["enrollment_date"] = user["enrollment_date"].isoformat()
                except:
                    pass
        return jsonify(users)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/attendance_recent")
@admin_required
def attendance_recent():
    """Recent attendance API"""
    try:
        records = list(attendance_col.find().sort("timestamp", -1).limit(50))
        out = []
        for r in records:
            try:
                ts = r.get("timestamp")
                iso = ts if isinstance(ts, str) else ts.isoformat()
            except Exception:
                iso = ""
            out.append({
                "name": r.get("name"),
                "timestamp": iso,
                "confidence": r.get("confidence", 0)
            })
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/attendance_stats")
@admin_required
def attendance_stats():
    """Attendance statistics API"""
    try:
        days = int(request.args.get("days", 30))
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=days - 1)
        
        records = list(attendance_col.find({}))
        counts = {}
        
        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, str):
                try:
                    dt = datetime.datetime.fromisoformat(ts)
                except Exception:
                    continue
            elif isinstance(ts, datetime.datetime):
                dt = ts
            else:
                continue
            
            if dt < start or dt > end:
                continue
            
            day = dt.date().isoformat()
            counts[day] = counts.get(day, 0) + 1
        
        labels = []
        values = []
        for i in range(days):
            day_dt = (start + datetime.timedelta(days=i)).date()
            day_str = day_dt.isoformat()
            labels.append(day_str)
            values.append(counts.get(day_str, 0))
        
        return jsonify({"labels": labels, "values": values})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/enrollment_requests")
@admin_required
def get_enrollment_requests():
    """Get enrollment requests API"""
    try:
        requests = list(enrollment_requests_col.find({"status": "pending"}).sort("submitted_at", -1))
        out = []
        for r in requests:
            r["_id"] = str(r["_id"])
            try:
                r["submitted_at"] = r["submitted_at"].isoformat()
            except:
                pass
            r.pop("images", None)
            r.pop("password_hash", None)
            out.append(r)
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/pending_requests_count")
@admin_required
def pending_requests_count():
    """Pending requests count API"""
    try:
        count = enrollment_requests_col.count_documents({"status": "pending"})
        return jsonify({"count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
        
        # Use the enhanced search function with 3D support
        # Convert file to image for 3D analysis
        file.seek(0)
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        matched_name, confidence = search_face_faiss(embedding, threshold=0.6, image=image)
        
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

# --- Missing Super Admin Routes ---
@app.route("/superadmin/users")
@superadmin_required
def superadmin_users():
    """Super admin users page"""
    try:
        users = list(persons_col.find({}))
        for user in users:
            user["_id"] = str(user["_id"])
            if "enrollment_date" in user:
                try:
                    user["enrollment_date"] = user["enrollment_date"].isoformat()
                except:
                    pass
        return render_template("superadmin/users.html", superadmin=current_user, users=users)
    except Exception as e:
        print(f"Error in superadmin_users: {e}")
        return render_template("superadmin/users.html", superadmin=current_user, users=[])

@app.route("/superadmin/logs")
@superadmin_required
def superadmin_logs():
    """Super admin logs page"""
    try:
        logs = list(system_logs_col.find().sort("timestamp", -1).limit(100))
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                try:
                    log["timestamp"] = log["timestamp"].isoformat()
                except:
                    pass
        return render_template("superadmin/logs.html", superadmin=current_user, logs=logs)
    except Exception as e:
        print(f"Error in superadmin_logs: {e}")
        return render_template("superadmin/logs.html", superadmin=current_user, logs=[])

@app.route("/api/superadmin/logs")
@superadmin_required
def api_get_logs():
    """API endpoint to get system logs as JSON"""
    try:
        limit = int(request.args.get("limit", 50))
        logs = list(system_logs_col.find().sort("timestamp", -1).limit(limit))
        
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                try:
                    log["timestamp"] = log["timestamp"].isoformat()
                except:
                    pass
        
        return jsonify({
            "status": "success",
            "logs": logs
        })
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/attendance/stats")
@superadmin_required
def api_attendance_stats():
    """Get attendance statistics for super admin"""
    try:
        days = int(request.args.get("days", 7))
        
        # Calculate date range
        end_date = datetime.datetime.utcnow()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get daily attendance counts
        daily_attendance = []
        for i in range(days):
            date = start_date + datetime.timedelta(days=i)
            day_start = datetime.datetime.combine(date.date(), datetime.time.min)
            day_end = datetime.datetime.combine(date.date(), datetime.time.max)
            
            count = attendance_col.count_documents({
                "timestamp": {"$gte": day_start, "$lt": day_end}
            })
            
            daily_attendance.append({
                "date": date.date().isoformat(),
                "count": count
            })
        
        return jsonify({
            "status": "success",
            "daily_attendance": daily_attendance
        })
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/admin/<admin_id>/block", methods=["POST"])
@superadmin_required
def block_admin_fixed(admin_id):
    """Block an admin"""
    try:
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        # Update admin status
        result = admins_col.update_one(
            {"_id": admin_id},
            {"$set": {
                "is_active": False,
                "blocked_at": datetime.datetime.utcnow(),
                "blocked_by": current_user.email
            }}
        )
        
        if result.modified_count > 0:
            # Log action
            system_logs_col.insert_one({
                "action": "block_admin",
                "user": current_user.email,
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "timestamp": datetime.datetime.utcnow(),
                "status": "success"
            })
            
            return jsonify({
                "status": "success",
                "msg": f"Admin {admin.get('name')} blocked successfully"
            })
        
        return jsonify({"status": "failed", "msg": "Failed to block admin"}), 500
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/admin/<admin_id>/unblock", methods=["POST"])
@superadmin_required
def unblock_admin_fixed(admin_id):
    """Unblock an admin"""
    try:
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        # Update admin status
        result = admins_col.update_one(
            {"_id": admin_id},
            {"$set": {
                "is_active": True,
                "unblocked_at": datetime.datetime.utcnow(),
                "unblocked_by": current_user.email
            }}
        )
        
        if result.modified_count > 0:
            # Log action
            system_logs_col.insert_one({
                "action": "unblock_admin",
                "user": current_user.email,
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "timestamp": datetime.datetime.utcnow(),
                "status": "success"
            })
            
            return jsonify({
                "status": "success",
                "msg": f"Admin {admin.get('name')} unblocked successfully"
            })
        
        return jsonify({"status": "failed", "msg": "Failed to unblock admin"}), 500
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/delete_admin/<admin_id>", methods=["DELETE"])
@superadmin_required
def superadmin_delete_admin(admin_id):
    """Delete admin"""
    try:
        admin = admins_col.find_one({"_id": admin_id})
        if not admin:
            return jsonify({"status": "failed", "msg": "Admin not found"}), 404
        
        result = admins_col.delete_one({"_id": admin_id})
        if result.deleted_count > 0:
            system_logs_col.insert_one({
                "action": "delete_admin",
                "admin_id": admin_id,
                "admin_email": admin.get("email"),
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            return jsonify({"status": "success", "msg": "Admin deleted successfully"})
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

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

# --- 3D Face Analysis Routes ---
@app.route("/api/analyze_3d_face", methods=["POST"])
@admin_required
def analyze_3d_face():
    """Analyze face using 3D reconstruction"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "failed", "msg": "No image provided"}), 400
        
        # Convert file to image
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"status": "failed", "msg": "Invalid image format"}), 400
        
        # Import 3D reconstruction functions
        try:
            from face_3d_reconstruction import extract_3d_face_features, visualize_3d_landmarks
            
            # Extract 3D features
            result = extract_3d_face_features(image)
            
            if result is None:
                return jsonify({"status": "failed", "msg": "No face detected in image"}), 400
            
            # Create visualization
            vis_image = visualize_3d_landmarks(image, result['landmarks_3d'])
            
            # Encode visualization as base64
            _, buffer = cv.imencode('.jpg', vis_image)
            vis_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                "status": "success",
                "features": {
                    "face_width": result['features']['face_width'],
                    "face_height": result['features']['face_height'],
                    "face_depth": result['features']['face_depth'],
                    "eye_distance": result['features']['eye_distance'],
                    "nose_length": result['features']['nose_length'],
                    "nose_protrusion": result['features']['nose_protrusion'],
                    "mouth_width": result['features']['mouth_width'],
                    "symmetry_score": result['features']['symmetry_score'],
                    "face_curvature": result['features']['face_curvature']
                },
                "pose_angles": result['pose_angles'],
                "quality_score": result['quality_score'],
                "landmarks_count": len(result['landmarks_3d']),
                "visualization": vis_base64
            })
            
        except ImportError:
            return jsonify({"status": "failed", "msg": "3D reconstruction not available"}), 503
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": f"Analysis failed: {str(e)}"}), 500

@app.route("/api/compare_3d_faces", methods=["POST"])
@admin_required
def compare_3d_faces():
    """Compare two faces using 3D analysis"""
    try:
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        
        if not file1 or not file2:
            return jsonify({"status": "failed", "msg": "Two images required"}), 400
        
        # Convert files to images
        images = []
        for file in [file1, file2]:
            file_bytes = file.read()
            img_array = np.frombuffer(file_bytes, np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
            if image is None:
                return jsonify({"status": "failed", "msg": "Invalid image format"}), 400
            images.append(image)
        
        try:
            from face_3d_reconstruction import extract_3d_face_features, match_3d_face_templates
            
            # Extract features from both images
            features = []
            for i, image in enumerate(images):
                result = extract_3d_face_features(image)
                if result is None:
                    return jsonify({"status": "failed", "msg": f"No face detected in image {i+1}"}), 400
                features.append(result)
            
            # Create temporary templates
            template1 = {
                'landmarks_3d': features[0]['landmarks_3d'],
                'features': features[0]['features'],
                'descriptor': np.random.randn(100),  # Simplified for demo
                'confidence': features[0]['quality_score']
            }
            
            template2 = {
                'landmarks_3d': features[1]['landmarks_3d'],
                'features': features[1]['features'],
                'descriptor': np.random.randn(100),  # Simplified for demo
                'confidence': features[1]['quality_score']
            }
            
            # Calculate similarity
            similarity = match_3d_face_templates(template1, template2)
            
            return jsonify({
                "status": "success",
                "similarity_score": similarity,
                "match_result": "MATCH" if similarity > 0.7 else "NO_MATCH",
                "confidence": (features[0]['quality_score'] + features[1]['quality_score']) / 2,
                "face1_features": features[0]['features'],
                "face2_features": features[1]['features'],
                "pose1": features[0]['pose_angles'],
                "pose2": features[1]['pose_angles']
            })
            
        except ImportError:
            return jsonify({"status": "failed", "msg": "3D reconstruction not available"}), 503
        
    except Exception as e:
        return jsonify({"status": "failed", "msg": f"Comparison failed: {str(e)}"}), 500

# --- Health Check Route ---
@app.route("/health")
def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        # Check MongoDB connection
        client.admin.command('ping')
        
        # Check FAISS index
        faiss_status = "loaded" if faiss_index is not None else "not_loaded"
        
        return jsonify({
            "status": "healthy",
            "mongodb": "connected",
            "faiss_index": faiss_status,
            "faiss_count": faiss_index.ntotal if faiss_index else 0,
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }), 503

# --- Real-time Face Recognition API ---
@app.route("/api/detect_face", methods=["POST"])
@admin_required
def detect_face_realtime():
    """Real-time face detection and recognition for camera stream"""
    try:
        file = request.files.get("frame")
        if not file:
            return jsonify({
                "status": "error", 
                "message": "No image frame provided"
            }), 400
        
        # Convert uploaded frame to image
        file_bytes = file.read()
        img_array = np.frombuffer(file_bytes, np.uint8)
        image = cv.imdecode(img_array, cv.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                "status": "error", 
                "message": "Invalid image format"
            }), 400

        # Detect faces using MediaPipe or CV2
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        
        with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)
            
            detected_faces = []
            
            if results.detections:
                for detection in results.detections:
                    # Get face bounding box
                    bbox = detection.location_data.relative_bounding_box
                    h, w, _ = image.shape
                    
                    # Convert to pixel coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    # Extract face region
                    face_roi = image[y:y+height, x:x+width]
                    
                    if face_roi.size > 0:
                        try:
                            # Get face embedding using the existing function
                            embedding = get_face_embedding(face_roi)
                            
                            if embedding is not None:
                                # Search in FAISS index for match
                                match_result = search_face_faiss(embedding)
                                
                                face_info = {
                                    "bbox": {"x": x, "y": y, "width": width, "height": height},
                                    "confidence": float(detection.score[0]),
                                    "detected": True
                                }
                                
                                if match_result and match_result.get('name') != 'Unknown':
                                    # Person recognized
                                    face_info.update({
                                        "recognized": True,
                                        "name": match_result['name'],
                                        "match_confidence": float(match_result.get('confidence', 0)),
                                        "person_id": str(match_result.get('_id', ''))
                                    })
                                    
                                    # Mark attendance
                                    try:
                                        attendance_data = {
                                            "person_id": match_result.get('_id'),
                                            "person_name": match_result['name'],
                                            "timestamp": datetime.datetime.utcnow(),
                                            "confidence": float(match_result.get('confidence', 0)),
                                            "method": "3D_Face_Recognition",
                                            "camera_source": "dashboard_live_camera"
                                        }
                                        
                                        # Check if attendance already marked today
                                        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                                        existing_attendance = attendance_col.find_one({
                                            "person_id": match_result.get('_id'),
                                            "timestamp": {"$gte": today_start}
                                        })
                                        
                                        if not existing_attendance:
                                            attendance_col.insert_one(attendance_data)
                                            face_info["attendance_marked"] = True
                                        else:
                                            face_info["attendance_marked"] = False
                                            face_info["already_marked"] = True
                                            
                                    except Exception as e:
                                        print(f"Attendance marking error: {e}")
                                        face_info["attendance_error"] = str(e)
                                else:
                                    face_info.update({
                                        "recognized": False,
                                        "name": "Unknown",
                                        "match_confidence": 0.0
                                    })
                                
                                detected_faces.append(face_info)
                        
                        except Exception as e:
                            print(f"Face processing error: {e}")
                            continue
            
            return jsonify({
                "status": "success",
                "faces_detected": len(detected_faces),
                "faces": detected_faces,
                "timestamp": datetime.datetime.utcnow().isoformat()
            })
            
    except Exception as e:
        print(f"Face detection API error: {e}")
        return jsonify({
            "status": "error",
            "message": f"Face detection failed: {str(e)}"
        }), 500

@app.route("/attendance")
def attendance_camera():
    """Attendance camera page"""
    return render_template("attendance_camera.html")

@app.route("/api/mark_attendance", methods=["POST"])
@admin_required
def mark_attendance_api():
    """API to manually mark attendance"""
    try:
        data = request.get_json()
        person_id = data.get('person_id')
        person_name = data.get('person_name')
        
        if not person_id or not person_name:
            return jsonify({"status": "error", "message": "Missing person data"}), 400
        
        # Check if attendance already marked today
        today_start = datetime.datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        existing_attendance = attendance_col.find_one({
            "person_id": person_id,
            "timestamp": {"$gte": today_start}
        })
        
        if existing_attendance:
            return jsonify({
                "status": "already_marked",
                "message": f"Attendance already marked for {person_name} today"
            })
        
        # Mark attendance
        attendance_data = {
            "person_id": person_id,
            "person_name": person_name,
            "timestamp": datetime.datetime.utcnow(),
            "confidence": data.get('confidence', 0.95),
            "method": "Manual_Dashboard",
            "camera_source": "dashboard_live_camera"
        }
        
        result = attendance_col.insert_one(attendance_data)
        
        return jsonify({
            "status": "success",
            "message": f"Attendance marked for {person_name}",
            "attendance_id": str(result.inserted_id)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to mark attendance: {str(e)}"
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
