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

from function import generate_camera_stream, get_face_embedding, continuous_learning_update, create_3d_template, extract_multi_vector_embeddings, ensemble_matching, search_face_faiss, rebuild_faiss_index

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
MONGODB_URI = os.environ.get("MONGODB_URI")
client = MongoClient(MONGODB_URI)

transactional_db = client["transactional_db"]
attendance_col = transactional_db["attendance"]

core = client['secure_db']
persons_col = core["persons"]
profile_col = core["profile"]
superadmins_col = core["superadmins"]  # Super Admin collection
admins_col = core["admins"]
users_col = core["users"]  
enrollment_requests_col = core["enrollment_requests"]
system_logs_col = core["system_logs"]  # System logs collection
cameras_col = core["cameras"]  # Camera management collection

# FAISS Vector Database Setup
EMBEDDING_DIM = 512
faiss_index = None
person_id_map = []

def initialize_faiss_index():
    global faiss_index, person_id_map
    index_path = "faiss_index.bin"
    map_path = "person_id_map.pkl"
    
    if os.path.exists(index_path) and os.path.exists(map_path):
        faiss_index = faiss.read_index(index_path)
        with open(map_path, 'rb') as f:
            person_id_map = pickle.load(f)
        print(f"Loaded FAISS index with {faiss_index.ntotal} faces")
    else:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        person_id_map = []
        print("Created new FAISS index")

def save_faiss_index():
    faiss.write_index(faiss_index, "faiss_index.bin")
    with open("person_id_map.pkl", 'wb') as f:
        pickle.dump(person_id_map, f)



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
    from bson.objectid import ObjectId
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
            flash('Access denied. Admin privileges required.', 'oops!')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Authentication Routes ---
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == 'superadmin':
            return redirect(url_for("superadmin_dashboard"))
        elif current_user.role == 'admin':
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        if current_user.role == 'superadmin':
            return redirect(url_for("superadmin_dashboard"))
        elif current_user.role == 'admin':
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    
    if request.method == "GET":
        return render_template("login.html")
    
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    remember = request.form.get("remember") == "on"
    
    if not email or not password:
        flash("Email and password required.", "danger")
        return redirect(url_for("login"))
    
    # Check super admin first
    superadmin = superadmins_col.find_one({"email": email})
    if superadmin:
        hashed = superadmin.get("password_hash")
        if hashed and check_password_hash(hashed, password):
            user = SuperAdminUser(superadmin)
            login_user(user, remember=remember)
            return redirect(url_for("superadmin_dashboard"))
    
    # Check admin
    admin = admins_col.find_one({"email": email})
    if admin:
        hashed = admin.get("password_hash")
        if hashed and check_password_hash(hashed, password):
            user = AdminUser(admin)
            login_user(user, remember=remember)
            return redirect(url_for("admin_dashboard"))
    
    # Check regular user
    user_doc = users_col.find_one({"email": email})
    if user_doc:
        if user_doc.get("status") == "blocked":
            flash("Your account has been blocked. Contact admin.", "danger")
            return redirect(url_for("login"))
        
        hashed = user_doc.get("password_hash")
        if hashed and check_password_hash(hashed, password):
            user = RegularUser(user_doc)
            login_user(user, remember=remember)
            return redirect(url_for("user_dashboard"))
    
    flash("Invalid credentials.", "danger")
    return redirect(url_for("login"))

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# Auth API Endpoints
@app.route("/auth/me")
@login_required
def auth_me():
    """Return current user information"""
    return jsonify({
        "status": "success",
        "user": {
            "email": current_user.email,
            "name": getattr(current_user, 'name', current_user.email),
            "role": current_user.role
        }
    })

@app.route("/api/auth/logout", methods=["POST"])
@login_required
def api_logout():
    """API logout endpoint"""
    logout_user()
    return jsonify({"status": "success", "msg": "Logged out successfully"})

@app.route("/create_admin", methods=["POST"])
def create_admin():
    secret = os.environ.get("CREATE_ADMIN_SECRET", "temporary_secret_for_local")
    provided = request.form.get("secret", "")
    
    if provided != secret:
        return jsonify({"status": "failed", "msg": "Invalid secret"}), 403
    
    name = request.form.get("name")
    email = request.form.get("email")
    password = request.form.get("password")
    
    if not (name and email and password):
        return jsonify({"status": "failed", "msg": "Missing fields"}), 400
    
    email = email.strip().lower()
    existing = admins_col.find_one({"email": email})
    if existing:
        return jsonify({"status": "failed", "msg": "Admin already exists"}), 400
    
    password_hash = generate_password_hash(password)
    doc = {
        "_id": str(datetime.datetime.utcnow().timestamp()) + "_" + email,
        "name": name,
        "email": email,
        "password_hash": password_hash,
        "profile_image": "",
        "created_at": datetime.datetime.utcnow()
    }
    admins_col.insert_one(doc)
    return jsonify({"status": "success", "msg": "Admin created"}), 201

# --- Admin Routes ---
@app.route("/admin/dashboard")
@admin_required
def admin_dashboard():
    """Admin Dashboard"""
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

@app.route("/superadmin/stats")
@superadmin_required
def superadmin_stats_api():
    """API endpoint for super admin statistics"""
    try:
        stats = {
            "total_users": persons_col.count_documents({"status": {"$ne": "blocked"}}),
            "total_admins": admins_col.count_documents({}),
            "pending_enrollments": enrollment_requests_col.count_documents({"status": "pending"}),
            "total_attendance_today": attendance_col.count_documents({
                "timestamp": {
                    "$gte": datetime.datetime.combine(datetime.date.today(), datetime.time.min),
                    "$lt": datetime.datetime.combine(datetime.date.today(), datetime.time.max)
                }
            }),
            "total_cameras": cameras_col.count_documents({}),
            "active_cameras": len(active_camera_streams),
            "system_health": "healthy"
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/admin/profile")
@admin_required
def admin_profile():
    profile_data = {
        "name": current_user.name,
        "email": current_user.email,
        "profile_image": current_user.profile_image,
        "role": "Administrator"
    }
    return render_template("admin_profile.html", profile=profile_data)

@app.route("/admin/update_profile", methods=["POST"])
@admin_required
def admin_update_profile():
    try:
        name = request.form.get("name")
        file = request.files.get("profile_image")
        
        update_data = {}
        if name:
            update_data["name"] = name
        
        if file:
            img_data = base64.b64encode(file.read()).decode('utf-8')
            update_data["profile_image"] = img_data
        
        admins_col.update_one(
            {"_id": current_user.id},
            {"$set": update_data}
        )
        
        flash("Profile updated successfully", "success")
        return redirect(url_for("admin_profile"))
    except Exception as e:
        flash("Error updating profile", "danger")
        return redirect(url_for("admin_profile"))

@app.route("/reg")
@admin_required
def reg():
    return render_template("registration_form.html")

@app.route("/enroll", methods=["POST"])
@limiter.limit("10 per hour")
@admin_required
def enroll():
    try:
        name = request.form.get("name", "").strip()
        department = request.form.get("department", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        files = request.files.getlist("files")
        
        if not name or not email or not password:
            return jsonify({"status": "failed", "msg": "Name, email and password required"}), 400
        
        if not files or len(files) < 5:
            return jsonify({"status": "failed", "msg": "Please upload at least 5 face images"}), 400
        
        # Check if email exists
        if users_col.find_one({"email": email}):
            return jsonify({"status": "failed", "msg": "Email already exists"}), 400
        
        print(f"[Enrollment] Processing {len(files)} images for {name}")
        
        # Extract multi-vector embeddings with quality scores
        embeddings_data = extract_multi_vector_embeddings(files)
        
        if len(embeddings_data) < 5:
            return jsonify({
                "status": "failed", 
                "msg": f"Only {len(embeddings_data)} valid faces detected. Need at least 5 high-quality images."
            }), 400
        
        print(f"[Enrollment] Extracted {len(embeddings_data)} valid embeddings")
        # --- FIX: compute formatted qualities first to avoid nested-quote f-string SyntaxError ---
        qualities = []
        for x in embeddings_data[:5]:
            q = x.get("quality")
            if q is None:
                qualities.append("N/A")
            else:
                try:
                    qualities.append(f"{q:.1f}")
                except (TypeError, ValueError):
                    qualities.append(str(q))
        print(f"[Enrollment] Quality scores: {qualities}")
        
        # Create 3D template with weighted centroid
        template = create_3d_template(embeddings_data)
        
        if template is None:
            return jsonify({"status": "failed", "msg": "Failed to create face template"}), 400
        
        # Use first high-quality image as profile picture
        profile_image = None
        for i, emb_data in enumerate(embeddings_data[:3]):
            try:
                file = files[emb_data['index']]
                file.seek(0)
                file_bytes = file.read()
                img_array = np.frombuffer(file_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                if image is not None:
                    # Resize for profile
                    image = cv.resize(image, (300, 300))
                    _, buffer = cv.imencode('.jpg', image)
                    profile_image = base64.b64encode(buffer).decode('utf-8')
                    break
            except Exception as e:
                print(f"[Profile Image Error] {e}")
                continue
        
        # Save to persons collection with 3D template
        persons_col.insert_one({
            "name": name,
            "embedding": pickle.dumps(template),
            "embedding_dim": int(template['centroid'].shape[0]),
            "photos_count": len(embeddings_data),
            "status": "active",
            "enrollment_date": datetime.datetime.now(),
            "template_type": "3D_multi_vector",
            "avg_quality": np.mean([x['quality'] for x in embeddings_data]),
            "update_count": 0
        })
        
        # Save profile
        profile_col.insert_one({
            "name": name,
            "department": department,
            "email": email,
            "phone": phone,
            "profile_image": profile_image or "",
            "registered_at": datetime.datetime.now()
        })
        
        # Create user account
        users_col.insert_one({
            "_id": str(datetime.datetime.utcnow().timestamp()) + "_" + email,
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "department": department,
            "phone": phone,
            "profile_image": profile_image or "",
            "status": "active",
            "created_at": datetime.datetime.now()
        })
        
        # Rebuild FAISS index with new user
        rebuild_faiss_index()
        
        print(f"[Enrollment] Successfully enrolled {name} with 3D template")
        
        return jsonify({
            "status": "success",
            "name": name,
            "photos_used": len(embeddings_data),
            "embedding_dim": int(template['centroid'].shape[0]),
            "avg_quality": round(np.mean([x['quality'] for x in embeddings_data]), 2),
            "template_type": "3D Multi-Vector"
        })
        
    except Exception as e:
        print("[Enrollment Fatal Error]", e)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/registration_request")
def registration_request():
    return render_template("registration_req.html")

@app.route("/submit_enrollment_request", methods=["POST"])
@limiter.limit("5 per hour")
def submit_enrollment_request():
    try:
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        files = request.files.getlist("files")
        
        if not name or not email or not password:
            return jsonify({"status": "failed", "msg": "Name, email and password are required"}), 400
        
        if not files or len(files) < 5:
            return jsonify({"status": "failed", "msg": "Please upload at least 5 face images"}), 400
        
        existing = enrollment_requests_col.find_one({"email": email, "status": "pending"})
        if existing:
            return jsonify({"status": "failed", "msg": "A pending request already exists"}), 400
        
        stored_images = []
        for idx, file in enumerate(files[:10]):
            try:
                file_bytes = file.read()
                img_b64 = base64.b64encode(file_bytes).decode('utf-8')
                stored_images.append(img_b64)
            except Exception as e:
                print(f"[Request Error] File {idx}: {e}")
                continue
        
        if len(stored_images) < 5:
            return jsonify({"status": "failed", "msg": "Failed to process images"}), 400
        
        request_doc = {
            "name": name,
            "email": email,
            "phone": phone,
            "password_hash": generate_password_hash(password),
            "images": stored_images,
            "status": "pending",
            "submitted_at": datetime.datetime.now(),
            "processed_at": None,
            "processed_by": None
        }
        
        enrollment_requests_col.insert_one(request_doc)
        
        return jsonify({
            "status": "success",
            "msg": "Enrollment request submitted successfully",
            "name": name
        })
    except Exception as e:
        print("[Request Submission Error]", e)
        return jsonify({"status": "failed", "msg": "Unexpected server error"}), 500

@app.route("/api/enrollment_requests")
@admin_required
def get_enrollment_requests():
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

@app.route("/api/enrollment_request/<request_id>")
@admin_required
def get_enrollment_request_detail(request_id):
    from bson.objectid import ObjectId
    try:
        req = enrollment_requests_col.find_one({"_id": ObjectId(request_id)})
        if req:
            req["_id"] = str(req["_id"])
            if "submitted_at" in req:
                req["submitted_at"] = req["submitted_at"].isoformat()
            req.pop("password_hash", None)
            return jsonify(req)
        return jsonify({"error": "Request not found"}), 404
    except:
        return jsonify({"error": "Invalid request ID"}), 400

@app.route("/api/approve_enrollment/<request_id>", methods=["POST"])
@admin_required
def approve_enrollment(request_id):
    """Approve enrollment request with 3D template creation"""
    from bson.objectid import ObjectId
    
    try:
        req = enrollment_requests_col.find_one({"_id": ObjectId(request_id)})
        if not req:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404
        
        if req["status"] != "pending":
            return jsonify({"status": "failed", "msg": "Request already processed"}), 400
        
        print(f"[Approval] Processing request for {req['name']}")
        
        # Decode and process images
        images = []
        for idx, img_b64 in enumerate(req["images"]):
            try:
                img_bytes = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                if image is not None:
                    images.append(image)
            except Exception as e:
                print(f"[Approval] Error decoding image {idx}: {e}")
                continue
        
        if len(images) < 5:
            return jsonify({"status": "failed", "msg": f"Only {len(images)} valid images. Need at least 5."}), 400
        
        # Extract multi-vector embeddings
        embeddings_data = extract_multi_vector_embeddings(images)
        
        if len(embeddings_data) < 5:
            return jsonify({"status": "failed", "msg": f"Only {len(embeddings_data)} valid faces detected"}), 400
        
        # Create 3D template
        template = create_3d_template(embeddings_data)
        
        if template is None:
            return jsonify({"status": "failed", "msg": "Failed to create face template"}), 400
        
        # Get profile image from best quality image
        profile_image = None
        best_quality_idx = embeddings_data[0]['index']
        try:
            img_bytes = base64.b64decode(req["images"][best_quality_idx])
            img_array = np.frombuffer(img_bytes, np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
            if image is not None:
                image = cv.resize(image, (300, 300))
                _, buffer = cv.imencode('.jpg', image)
                profile_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[Profile Image Error] {e}")
        
        department = request.form.get("department", "")
        
        # Save to persons collection with 3D template
        persons_col.insert_one({
            "name": req["name"],
            "embedding": pickle.dumps(template),
            "embedding_dim": int(template['centroid'].shape[0]),
            "photos_count": len(embeddings_data),
            "status": "active",
            "enrollment_date": datetime.datetime.now(),
            "template_type": "3D_multi_vector",
            "avg_quality": np.mean([x['quality'] for x in embeddings_data]),
            "update_count": 0
        })
        
        # Save profile
        profile_col.insert_one({
            "name": req["name"],
            "department": department,
            "email": req["email"],
            "phone": req["phone"],
            "profile_image": profile_image or "",
            "registered_at": datetime.datetime.now()
        })
        
        # Create user account
        users_col.insert_one({
            "_id": str(datetime.datetime.utcnow().timestamp()) + "_" + req["email"],
            "name": req["name"],
            "email": req["email"],
            "password_hash": req["password_hash"],
            "department": department,
            "phone": req["phone"],
            "profile_image": profile_image or "",
            "status": "active",
            "created_at": datetime.datetime.now()
        })
        
        # Update request status
        enrollment_requests_col.update_one(
            {"_id": ObjectId(request_id)},
            {"$set": {
                "status": "approved",
                "processed_at": datetime.datetime.now(),
                "processed_by": current_user.email
            }}
        )
        
        # Rebuild FAISS index
        rebuild_faiss_index()
        
        print(f"[Approval] Successfully approved {req['name']} with 3D template")
        
        return jsonify({
            "status": "success",
            "msg": f"{req['name']} has been enrolled successfully",
            "photos_used": len(embeddings_data),
            "avg_quality": round(np.mean([x['quality'] for x in embeddings_data]), 2)
        })
        
    except Exception as e:
        print("[Approval Fatal Error]", e)
        import traceback
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/reject_enrollment/<request_id>", methods=["POST"])
@admin_required
def reject_enrollment(request_id):
    from bson.objectid import ObjectId
    
    try:
        reason = request.form.get("reason", "Not specified")
        
        result = enrollment_requests_col.update_one(
            {"_id": ObjectId(request_id)},
            {"$set": {
                "status": "rejected",
                "rejection_reason": reason,
                "processed_at": datetime.datetime.now(),
                "processed_by": current_user.email
            }}
        )
        
        if result.modified_count > 0:
            return jsonify({"status": "success", "msg": "Request rejected"})
        return jsonify({"status": "failed", "msg": "Request not found"}), 404
    except Exception as e:
        print("[Rejection Error]", e)
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/pending_requests_count")
@admin_required
def pending_requests_count():
    count = enrollment_requests_col.count_documents({"status": "pending"})
    return jsonify({"count": count})

@app.route("/api/attendance_recent")
@admin_required
def attendance_recent():
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

@app.route("/api/attendance_stats")
@admin_required
def attendance_stats():
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

@app.route("/list_users")
@admin_required
def list_users():
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

@app.route("/api/block_user", methods=["POST"])
@admin_required
def block_user():
    """Block a user from the system"""
    try:
        # Accept both form data and JSON
        if request.is_json:
            data = request.get_json()
            name = data.get("name")
            user_id = data.get("user_id")
        else:
            name = request.form.get("name")
            user_id = request.form.get("user_id")
        
        if not name and not user_id:
            return jsonify({"status": "failed", "msg": "Name or user_id required"}), 400
        
        # Find user by name or ID
        query = {"name": name} if name else {"_id": ObjectId(user_id)}
        
        # Update person status
        result = persons_col.update_one(query, {"$set": {"status": "blocked"}})
        users_col.update_one(query, {"$set": {"status": "blocked"}})
        
        if result.matched_count == 0:
            return jsonify({"status": "failed", "msg": "User not found"}), 404
        
        # Log the action
        system_logs_col.insert_one({
            "action": "block_user",
            "user_name": name or user_id,
            "admin_email": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        # Rebuild FAISS index in background (non-blocking)
        import threading
        threading.Thread(target=rebuild_faiss_index, daemon=True).start()
        
        return jsonify({
            "status": "success",
            "msg": f"{name or 'User'} has been blocked successfully"
        })
    except Exception as e:
        print(f"[Block User Error] {e}")
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/unblock_user", methods=["POST"])
@admin_required
def unblock_user():
    """Unblock a user"""
    try:
        # Accept both form data and JSON
        if request.is_json:
            data = request.get_json()
            name = data.get("name")
            user_id = data.get("user_id")
        else:
            name = request.form.get("name")
            user_id = request.form.get("user_id")
        
        if not name and not user_id:
            return jsonify({"status": "failed", "msg": "Name or user_id required"}), 400
        
        # Find user by name or ID
        query = {"name": name} if name else {"_id": ObjectId(user_id)}
        
        # Update person status
        result = persons_col.update_one(query, {"$set": {"status": "active"}})
        users_col.update_one(query, {"$set": {"status": "active"}})
        
        if result.matched_count == 0:
            return jsonify({"status": "failed", "msg": "User not found"}), 404
        
        # Log the action
        system_logs_col.insert_one({
            "action": "unblock_user",
            "user_name": name or user_id,
            "admin_email": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        # Rebuild FAISS index in background (non-blocking)
        import threading
        threading.Thread(target=rebuild_faiss_index, daemon=True).start()
        
        return jsonify({
            "status": "success",
            "msg": f"{name or 'User'} has been unblocked successfully"
        })
    except Exception as e:
        print(f"[Unblock User Error] {e}")
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/system_stats")
@admin_required
def system_stats():
    """Get system-wide statistics including continuous learning metrics"""
    try:
        # Count users with 3D templates
        users_with_3d = persons_col.count_documents({"template_type": "3D_multi_vector"})
        total_users = persons_col.count_documents({})
        
        # Get average update count (continuous learning activity)
        pipeline = [
            {"$group": {
                "_id": None,
                "avg_updates": {"$avg": "$update_count"},
                "total_updates": {"$sum": "$update_count"}
            }}
        ]
        learning_stats = list(persons_col.aggregate(pipeline))
        
        # Get average quality scores
        quality_pipeline = [
            {"$match": {"avg_quality": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "avg_quality": {"$avg": "$avg_quality"}
            }}
        ]
        quality_stats = list(persons_col.aggregate(quality_pipeline))
        
        return jsonify({
            "total_users": total_users,
            "users_with_3d_template": users_with_3d,
            "3d_template_percentage": round((users_with_3d / total_users * 100) if total_users > 0 else 0, 2),
            "avg_updates_per_user": round(learning_stats[0]['avg_updates'], 2) if learning_stats else 0,
            "total_continuous_updates": learning_stats[0]['total_updates'] if learning_stats else 0,
            "avg_enrollment_quality": round(quality_stats[0]['avg_quality'], 2) if quality_stats else 0,
            "faiss_index_size": faiss_index.ntotal if faiss_index else 0
        })
    except Exception as e:
        print(f"[System Stats Error] {e}")
        return jsonify({"error": str(e)}), 500

# --- Mobile Camera Routes ---
@app.route("/mobile/capture")
def mobile_capture():
    """Mobile camera capture page - accessible without login for quick attendance"""
    return render_template("mobile_capture.html")

@app.route("/api/mobile/mark_attendance", methods=["POST"])
@limiter.limit("20 per hour")
def mobile_mark_attendance():
    """Mark attendance from mobile camera capture"""
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "failed", "msg": "No image provided"}), 400
        
        # Get face embedding
        embedding = get_face_embedding(file)
        if embedding is None:
            return jsonify({"status": "failed", "msg": "No face detected. Please ensure your face is clearly visible and try again."}), 400
        
        # Search in FAISS
        if face_index is None or face_index.ntotal == 0:
            return jsonify({"status": "failed", "msg": "System not initialized. Please contact administrator."}), 500
        
        D, I = face_index.search(np.array([embedding]), k=1)
        
        MATCH_THRESHOLD = 0.6
        if len(I[0]) == 0 or D[0][0] > MATCH_THRESHOLD:
            return jsonify({
                "status": "failed",
                "msg": "Face not recognized. Please enroll first or contact administrator."
            }), 404
        
        person_id = index_to_person_id.get(I[0][0])
        if not person_id:
            return jsonify({"status": "failed", "msg": "Person not found in database"}), 404
        
        # Get person details
        person = persons_col.find_one({"_id": person_id})
        if not person:
            return jsonify({"status": "failed", "msg": "Person data not found"}), 404
        
        # Check if already marked today
        today_start = datetime.datetime.combine(datetime.date.today(), datetime.time.min)
        existing = attendance_col.find_one({
            "person_id": person_id,
            "timestamp": {"$gte": today_start}
        })
        
        if existing:
            return jsonify({
                "status": "success",
                "message": "Attendance already marked for today",
                "person_name": person.get("name", "Unknown"),
                "time": existing["timestamp"].strftime("%I:%M %p"),
                "confidence": round(existing.get("confidence", 0) * 100, 2)
            })
        
        # Mark attendance
        confidence = float(1.0 - D[0][0])
        attendance_col.insert_one({
            "person_id": person_id,
            "person_name": person.get("name", "Unknown"),
            "email": person.get("email", ""),
            "department": person.get("department", ""),
            "timestamp": datetime.datetime.utcnow(),
            "confidence": confidence,
            "device": "mobile",
            "user_agent": request.headers.get("User-Agent", ""),
            "location": request.headers.get("X-Forwarded-For", request.remote_addr)
        })
        
        # Log attendance event
        system_logs_col.insert_one({
            "action": "mobile_attendance",
            "person_id": str(person_id),
            "person_name": person.get("name", "Unknown"),
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
        import traceback
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": "Server error. Please try again or contact administrator."}), 500

# --- User Routes ---
@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if current_user.role == 'admin':
        return redirect(url_for("admin_dashboard"))
    if current_user.role == 'superadmin':
        return redirect(url_for("superadmin_dashboard"))
    return render_template("user_dashboard_new.html", user=current_user)

@app.route("/user/profile")
@login_required
def user_profile():
    profile = profile_col.find_one({"email": current_user.email})
    return render_template("user_profile.html", profile=profile, user=current_user)

@app.route("/user/update_profile", methods=["POST"])
@login_required
def user_update_profile():
    try:
        phone = request.form.get("phone")
        file = request.files.get("profile_image")
        
        update_data = {}
        if phone:
            update_data["phone"] = phone
        
        if file:
            img_data = base64.b64encode(file.read()).decode('utf-8')
            update_data["profile_image"] = img_data
        
        if update_data:
            profile_col.update_one({"email": current_user.email}, {"$set": update_data})
            users_col.update_one({"email": current_user.email}, {"$set": update_data})
        
        flash("Profile updated successfully", "success")
        return redirect(url_for("user_profile"))
    except Exception as e:
        flash("Error updating profile", "danger")
        return redirect(url_for("user_profile"))

@app.route("/api/user/attendance_history")
@login_required
def user_attendance_history():
    records = list(attendance_col.find({"name": current_user.name}).sort("timestamp", -1).limit(100))
    out = []
    for r in records:
        try:
            ts = r.get("timestamp")
            iso = ts if isinstance(ts, str) else ts.isoformat()
        except Exception:
            iso = ""
        out.append({
            "timestamp": iso,
            "confidence": r.get("confidence", 0)
        })
    return jsonify(out)

@app.route("/api/user/attendance_stats")
@login_required
def user_attendance_stats():
    days = int(request.args.get("days", 30))
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days - 1)
    
    records = list(attendance_col.find({"name": current_user.name}))
    
    total_days = 0
    present_days = set()
    
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
        
        if dt >= start and dt <= end:
            present_days.add(dt.date().isoformat())
    
    return jsonify({
        "present_days": len(present_days),
        "total_days": days,
        "percentage": round((len(present_days) / days) * 100, 2)
    })

@app.route("/video_feed")
@admin_required
def video_feed():
    return Response(generate_camera_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/mark_attendance", methods=["POST"])
@limiter.limit("100 per hour")
@admin_required
def mark_attendance():
    file = request.files.get("file")
    if not file:
        return jsonify({"status": "failed", "msg": "No file uploaded"}), 400
    
    embedding = get_face_embedding(file)
    if embedding is None:
        return jsonify({"status": "failed", "msg": "No face detected"}), 400
    
    matched_name, confidence = search_face_faiss(embedding, threshold=0.6)
    
    if matched_name:
        person_doc = persons_col.find_one({"name": matched_name})
        if person_doc and person_doc.get("status") == "blocked":
            return jsonify({"status": "failed", "msg": f"{matched_name} is blocked"}), 403
        
        template_type = person_doc.get("template_type", "Standard") if person_doc else "Standard"
        
        attendance_col.insert_one({
            "name": matched_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "confidence": confidence,
            "template_type": template_type,
            "method": "manual"
        })
        return jsonify({
            "status": "success",
            "msg": f"Attendance marked for {matched_name}",
            "name": matched_name,
            "score": round(confidence, 3),
            "template_type": template_type
        })
    
    return jsonify({"status": "failed", "msg": "No match found"}), 404

# --- Super Admin Routes ---
@app.route("/superadmin/dashboard")
@superadmin_required
def superadmin_dashboard():
    """Super Admin Dashboard"""
    total_admins = admins_col.count_documents({})
    total_users = persons_col.count_documents({})
    total_attendance = attendance_col.count_documents({})
    pending_requests = enrollment_requests_col.count_documents({"status": "pending"})
    
    return render_template("superadmin/dashboard.html", 
                         superadmin=current_user,
                         total_admins=total_admins,
                         total_users=total_users,
                         total_attendance=total_attendance,
                         pending_requests=pending_requests)

@app.route("/superadmin/admins")
@superadmin_required
def superadmin_admins():
    """Manage Admins"""
    admins = list(admins_col.find({}, {"password_hash": 0}))
    for admin in admins:
        admin["_id"] = str(admin["_id"])
        if "created_at" in admin:
            try:
                admin["created_at"] = admin["created_at"].isoformat()
            except:
                pass
    return render_template("superadmin/admins.html", superadmin=current_user, admins=admins)

@app.route("/api/superadmin/create_admin", methods=["POST"])
@superadmin_required
def superadmin_create_admin():
    """Create new admin"""
    try:
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        
        if not name or not email or not password:
            return jsonify({"status": "failed", "msg": "All fields required"}), 400
        
        if admins_col.find_one({"email": email}):
            return jsonify({"status": "failed", "msg": "Admin already exists"}), 400
        
        admin_id = f"{datetime.datetime.utcnow().timestamp()}_{email}"
        admin_doc = {
            "_id": admin_id,
            "name": name,
            "email": email,
            "password_hash": generate_password_hash(password),
            "profile_image": "",
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email
        }
        admins_col.insert_one(admin_doc)
        
        # Log action
        system_logs_col.insert_one({
            "action": "create_admin",
            "admin_email": email,
            "performed_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "msg": f"Admin {name} created successfully"})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/delete_admin/<admin_id>", methods=["DELETE"])
@superadmin_required
def superadmin_delete_admin(admin_id):
    """Delete admin"""
    try:
        result = admins_col.delete_one({"_id": admin_id})
        if result.deleted_count > 0:
            system_logs_col.insert_one({
                "action": "delete_admin",
                "admin_id": admin_id,
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            return jsonify({"status": "success", "msg": "Admin deleted"})
        return jsonify({"status": "failed", "msg": "Admin not found"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/superadmin/users")
@superadmin_required
def superadmin_users():
    """View all users"""
    users = list(persons_col.find({}))
    for user in users:
        user["_id"] = str(user["_id"])
        if "enrollment_date" in user:
            try:
                user["enrollment_date"] = user["enrollment_date"].isoformat()
            except:
                pass
    return render_template("superadmin/users.html", superadmin=current_user, users=users)

@app.route("/superadmin/cameras")
@superadmin_required
def superadmin_cameras():
    """Camera management"""
    return render_template("superadmin/cameras.html", superadmin=current_user)

@app.route("/superadmin/logs")
@superadmin_required
def superadmin_logs():
    """System logs"""
    logs = list(system_logs_col.find().sort("timestamp", -1).limit(100))
    for log in logs:
        log["_id"] = str(log["_id"])
        if "timestamp" in log:
            try:
                log["timestamp"] = log["timestamp"].isoformat()
            except:
                pass
    return render_template("superadmin/logs.html", superadmin=current_user, logs=logs)

@app.route("/api/superadmin/system_stats")
@superadmin_required
def superadmin_system_stats():
    """Get comprehensive system statistics"""
    try:
        stats = {
            "total_superadmins": superadmins_col.count_documents({}),
            "total_admins": admins_col.count_documents({}),
            "total_users": persons_col.count_documents({}),
            "active_users": persons_col.count_documents({"status": "active"}),
            "blocked_users": persons_col.count_documents({"status": "blocked"}),
            "total_attendance": attendance_col.count_documents({}),
            "pending_requests": enrollment_requests_col.count_documents({"status": "pending"}),
            "faiss_index_size": faiss_index.ntotal if faiss_index else 0,
            "recent_logs": system_logs_col.count_documents({})
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/superadmin/users")
@superadmin_required
def api_get_all_users():
    """API endpoint to get all users as JSON"""
    try:
        users = list(persons_col.find({}))
        for user in users:
            user["_id"] = str(user["_id"])
            if "created_at" in user:
                try:
                    user["created_at"] = user["created_at"].isoformat()
                except:
                    pass
            if "last_attendance" in user:
                try:
                    user["last_attendance"] = user["last_attendance"].isoformat()
                except:
                    pass
        return jsonify({"status": "success", "users": users})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/logs")
@superadmin_required
def api_get_logs():
    """API endpoint to get system logs as JSON"""
    try:
        per_page = int(request.args.get("per_page", 50))
        page = int(request.args.get("page", 1))
        
        skip = (page - 1) * per_page
        logs = list(system_logs_col.find().sort("timestamp", -1).skip(skip).limit(per_page))
        total = system_logs_col.count_documents({})
        
        for log in logs:
            log["_id"] = str(log["_id"])
            if "timestamp" in log:
                try:
                    log["timestamp"] = log["timestamp"].isoformat()
                except:
                    pass
        
        return jsonify({
            "status": "success",
            "logs": logs,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/superadmin/attendance/live")
@superadmin_required
def get_live_attendance():
    """Get recent attendance records for live monitoring"""
    try:
        limit = int(request.args.get("limit", 10))
        attendance_records = list(
            attendance_col.find()
            .sort("timestamp", -1)
            .limit(limit)
        )
        
        for record in attendance_records:
            record["_id"] = str(record["_id"])
            if "person_id" in record:
                record["person_id"] = str(record["person_id"])
            if "timestamp" in record:
                try:
                    record["timestamp"] = record["timestamp"].isoformat()
                except:
                    pass
        
        return jsonify({"status": "success", "attendance": attendance_records})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/superadmin/attendance/stats")
@superadmin_required
def get_attendance_stats():
    """Get attendance statistics for charts"""
    try:
        days = int(request.args.get("days", 30))
        
        # Get daily attendance for last N days
        from datetime import timedelta
        today = datetime.date.today()
        daily_attendance = []
        
        for i in range(days):
            date = today - timedelta(days=days - i - 1)
            day_start = datetime.datetime.combine(date, datetime.time.min)
            day_end = datetime.datetime.combine(date, datetime.time.max)
            
            count = attendance_col.count_documents({
                "timestamp": {"$gte": day_start, "$lt": day_end}
            })
            
            daily_attendance.append({
                "date": date.isoformat(),
                "count": count
            })
        
        return jsonify({
            "status": "success",
            "daily_attendance": daily_attendance
        })
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

# ============================================================================
# CAMERA MANAGEMENT SYSTEM
# ============================================================================

# Global dictionary to track active camera streams
active_camera_streams = {}
camera_stream_locks = {}

@app.route("/api/superadmin/cameras", methods=["GET"])
@superadmin_required
def get_all_cameras():
    """Get all cameras with their status"""
    try:
        cameras = list(cameras_col.find({}))
        for cam in cameras:
            cam["_id"] = str(cam["_id"])
            if "created_at" in cam:
                try:
                    cam["created_at"] = cam["created_at"].isoformat()
                except:
                    pass
            if "last_seen" in cam:
                try:
                    cam["last_seen"] = cam["last_seen"].isoformat()
                except:
                    pass
            # Add stream status
            cam["stream_active"] = cam["_id"] in active_camera_streams
        return jsonify({"status": "success", "cameras": cameras})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/cameras/cameras", methods=["GET"])
@superadmin_required
def get_cameras_alias():
    """Alias endpoint for backward compatibility"""
    return get_all_cameras()

@app.route("/api/superadmin/cameras", methods=["POST"])
@limiter.limit("20 per hour")
@superadmin_required
def create_camera():
    """Create new camera configuration"""
    try:
        name = request.form.get("name", "").strip()
        source_type = request.form.get("source_type", "opencv")  # opencv/int or stream/url
        source = request.form.get("source", "0")  # int for device index or URL
        
        # Optional configuration
        auth_username = request.form.get("auth_username", "")
        auth_password = request.form.get("auth_password", "")
        fps = int(request.form.get("fps", 30))
        resolution_width = int(request.form.get("resolution_width", 640))
        resolution_height = int(request.form.get("resolution_height", 480))
        
        if not name:
            return jsonify({"status": "failed", "msg": "Camera name required"}), 400
        
        # Check if camera name exists
        if cameras_col.find_one({"name": name}):
            return jsonify({"status": "failed", "msg": "Camera name already exists"}), 400
        
        # Parse source (convert to int if it's a device index)
        if source_type == "opencv":
            try:
                source = int(source)
            except:
                source = 0
        
        camera_id = f"{datetime.datetime.utcnow().timestamp()}_{name.replace(' ', '_')}"
        
        camera_doc = {
            "_id": camera_id,
            "name": name,
            "source_type": source_type,
            "source": source,
            "auth": {
                "username": auth_username,
                "password": auth_password
            } if auth_username else None,
            "config": {
                "fps": fps,
                "resolution": {"width": resolution_width, "height": resolution_height},
                "detection_roi": None,  # Can be set later for specific area detection
                "fallback_timeout": 30  # seconds before considering stream dead
            },
            "enabled": True,
            "last_seen": None,
            "created_at": datetime.datetime.utcnow(),
            "created_by": current_user.email,
            "stream_url": f"/camera_feed/{camera_id}"
        }
        
        cameras_col.insert_one(camera_doc)
        
        # Log action
        system_logs_col.insert_one({
            "action": "create_camera",
            "camera_id": camera_id,
            "camera_name": name,
            "performed_by": current_user.email,
            "timestamp": datetime.datetime.utcnow()
        })
        
        return jsonify({"status": "success", "msg": f"Camera '{name}' created", "camera_id": camera_id})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/cameras/<camera_id>", methods=["PUT"])
@superadmin_required
def update_camera(camera_id):
    """Update camera configuration"""
    try:
        camera = cameras_col.find_one({"_id": camera_id})
        if not camera:
            return jsonify({"status": "failed", "msg": "Camera not found"}), 404
        
        update_data = {}
        
        if "name" in request.form:
            update_data["name"] = request.form.get("name").strip()
        if "source_type" in request.form:
            update_data["source_type"] = request.form.get("source_type")
        if "source" in request.form:
            source = request.form.get("source")
            if update_data.get("source_type", camera["source_type"]) == "opencv":
                try:
                    source = int(source)
                except:
                    source = 0
            update_data["source"] = source
        if "enabled" in request.form:
            update_data["enabled"] = request.form.get("enabled").lower() == "true"
        
        # Update config if provided
        if any(k in request.form for k in ["fps", "resolution_width", "resolution_height"]):
            config = camera.get("config", {})
            if "fps" in request.form:
                config["fps"] = int(request.form.get("fps"))
            if "resolution_width" in request.form or "resolution_height" in request.form:
                resolution = config.get("resolution", {})
                if "resolution_width" in request.form:
                    resolution["width"] = int(request.form.get("resolution_width"))
                if "resolution_height" in request.form:
                    resolution["height"] = int(request.form.get("resolution_height"))
                config["resolution"] = resolution
            update_data["config"] = config
        
        if update_data:
            update_data["updated_at"] = datetime.datetime.utcnow()
            update_data["updated_by"] = current_user.email
            cameras_col.update_one({"_id": camera_id}, {"$set": update_data})
            
            # Log action
            system_logs_col.insert_one({
                "action": "update_camera",
                "camera_id": camera_id,
                "changes": list(update_data.keys()),
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
        
        return jsonify({"status": "success", "msg": "Camera updated"})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/cameras/<camera_id>", methods=["DELETE"])
@superadmin_required
def delete_camera(camera_id):
    """Delete camera"""
    try:
        # Stop stream if active
        if camera_id in active_camera_streams:
            stop_camera_stream(camera_id)
        
        result = cameras_col.delete_one({"_id": camera_id})
        if result.deleted_count > 0:
            # Log action
            system_logs_col.insert_one({
                "action": "delete_camera",
                "camera_id": camera_id,
                "performed_by": current_user.email,
                "timestamp": datetime.datetime.utcnow()
            })
            return jsonify({"status": "success", "msg": "Camera deleted"})
        return jsonify({"status": "failed", "msg": "Camera not found"}), 404
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/cameras/<camera_id>/start", methods=["POST"])
@superadmin_required
def start_camera_stream_endpoint(camera_id):
    """Start camera streaming"""
    try:
        camera = cameras_col.find_one({"_id": camera_id})
        if not camera:
            return jsonify({"status": "failed", "msg": "Camera not found"}), 404
        
        if not camera.get("enabled"):
            return jsonify({"status": "failed", "msg": "Camera is disabled"}), 400
        
        if camera_id in active_camera_streams:
            return jsonify({"status": "success", "msg": "Camera stream already active"})
        
        # Start stream in background thread
        import threading
        thread = threading.Thread(target=initialize_camera_stream, args=(camera_id, camera))
        thread.daemon = True
        thread.start()
        
        return jsonify({"status": "success", "msg": "Camera stream started", "stream_url": f"/camera_feed/{camera_id}"})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/api/superadmin/cameras/<camera_id>/stop", methods=["POST"])
@superadmin_required
def stop_camera_stream_endpoint(camera_id):
    """Stop camera streaming"""
    try:
        stop_camera_stream(camera_id)
        return jsonify({"status": "success", "msg": "Camera stream stopped"})
    except Exception as e:
        return jsonify({"status": "failed", "msg": str(e)}), 500

def initialize_camera_stream(camera_id, camera):
    """Initialize camera capture in background thread"""
    try:
        source = camera["source"]
        config = camera.get("config", {})
        
        # Open camera
        cap = cv.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"[Camera {camera_id}] Failed to open camera source: {source}")
            return
        
        # Set resolution if specified
        resolution = config.get("resolution", {})
        if resolution.get("width"):
            cap.set(cv.CAP_PROP_FRAME_WIDTH, resolution["width"])
        if resolution.get("height"):
            cap.set(cv.CAP_PROP_FRAME_HEIGHT, resolution["height"])
        if config.get("fps"):
            cap.set(cv.CAP_PROP_FPS, config["fps"])
        
        # Store capture object
        import threading
        active_camera_streams[camera_id] = cap
        camera_stream_locks[camera_id] = threading.Lock()
        
        # Update last_seen
        cameras_col.update_one(
            {"_id": camera_id},
            {"$set": {"last_seen": datetime.datetime.utcnow()}}
        )
        
        print(f"[Camera {camera_id}] Stream initialized: {camera['name']}")
    except Exception as e:
        print(f"[Camera {camera_id}] Error initializing stream: {e}")

def stop_camera_stream(camera_id):
    """Stop camera stream and release resources"""
    try:
        if camera_id in active_camera_streams:
            cap = active_camera_streams[camera_id]
            cap.release()
            del active_camera_streams[camera_id]
            
        if camera_id in camera_stream_locks:
            del camera_stream_locks[camera_id]
            
        print(f"[Camera {camera_id}] Stream stopped")
    except Exception as e:
        print(f"[Camera {camera_id}] Error stopping stream: {e}")

@app.route("/camera_feed/<camera_id>")
@superadmin_required
def camera_feed(camera_id):
    """Streaming endpoint for camera feed"""
    def generate_frames(camera_id):
        """Generate frames from camera"""
        while camera_id in active_camera_streams:
            try:
                cap = active_camera_streams[camera_id]
                lock = camera_stream_locks.get(camera_id)
                
                if lock:
                    with lock:
                        success, frame = cap.read()
                else:
                    success, frame = cap.read()
                
                if not success:
                    print(f"[Camera {camera_id}] Failed to read frame")
                    break
                
                # Encode frame as JPEG
                ret, buffer = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                
                # Update last_seen timestamp periodically
                cameras_col.update_one(
                    {"_id": camera_id},
                    {"$set": {"last_seen": datetime.datetime.utcnow()}}
                )
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"[Camera {camera_id}] Frame generation error: {e}")
                break
        
        # Clean up when stream ends
        stop_camera_stream(camera_id)
    
    return Response(generate_frames(camera_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# Health check endpoint for Railway
@app.route("/health")
def health_check():
    """Health check endpoint for deployment platforms"""
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

# Auto-create super admin on startup
def create_default_superadmin():
    """Create default super admin if none exists"""
    try:
        superadmin_email = os.environ.get("SUPERADMIN_EMAIL", "superadmin@admin.com")
        superadmin_password = os.environ.get("SUPERADMIN_PASSWORD", "SuperAdmin@123")
        
        existing = superadmins_col.find_one({"email": superadmin_email})
        if not existing:
            superadmin_id = f"{datetime.datetime.utcnow().timestamp()}_{superadmin_email}"
            superadmin_doc = {
                "_id": superadmin_id,
                "name": "Super Administrator",
                "email": superadmin_email,
                "password_hash": generate_password_hash(superadmin_password),
                "profile_image": "",
                "created_at": datetime.datetime.utcnow()
            }
            superadmins_col.insert_one(superadmin_doc)
            print(f"✅ Super Admin created: {superadmin_email}")
            print(f"   Password: {superadmin_password}")
        else:
            print(f"✅ Super Admin exists: {existing['email']}")
    except Exception as e:
        print(f"⚠️ Super Admin creation error: {e}")

# Auto-create admin on startup
def create_default_admin():
    """Create default admin if none exists"""
    try:
        admin_email = os.environ.get("ADMIN_EMAIL", "admin@admin.com")
        admin_password = os.environ.get("ADMIN_PASSWORD", "password123")
        
        existing = admins_col.find_one({"email": admin_email})
        if not existing:
            admin_id = f"{datetime.datetime.utcnow().timestamp()}_{admin_email}"
            admin_doc = {
                "_id": admin_id,
                "name": "System Admin",
                "email": admin_email,
                "password_hash": generate_password_hash(admin_password),
                "profile_image": "",
                "created_at": datetime.datetime.utcnow()
            }
            admins_col.insert_one(admin_doc)
            print(f"✅ Admin created: {admin_email}")
        else:
            print(f"✅ Admin exists: {existing['email']}")
    except Exception as e:
        print(f"⚠️ Admin creation error: {e}")

# Initialize on startup with better error handling
def initialize_app():
    """Initialize application components with error handling"""
    print("[Startup] Initializing application...")
    print("="*60)
    
    try:
        # Test MongoDB connection
        client.admin.command('ping')
        print("✅ MongoDB connected")
    except Exception as e:
        print(f"⚠️ MongoDB connection error: {e}")
        print("Application will continue but database operations may fail")
    
    try:
        # Initialize FAISS index
        initialize_faiss_index()
        print("✅ FAISS index initialized")
    except Exception as e:
        print(f"⚠️ FAISS initialization error: {e}")
    
    try:
        # Create default super admin
        create_default_superadmin()
    except Exception as e:
        print(f"⚠️ Super Admin creation error: {e}")
    
    try:
        # Create default admin
        create_default_admin()
    except Exception as e:
        print(f"⚠️ Admin creation error: {e}")
    
    print("="*60)
    print("[Startup] Application initialization complete")
    print("")
    print("DEFAULT CREDENTIALS:")
    print(f"   Super Admin: superadmin@admin.com / SuperAdmin@123")
    print(f"   Admin: admin@admin.com / password123")
    print("   ⚠️  CHANGE THESE PASSWORDS IMMEDIATELY!")
    print("="*60)

# Run initialization
initialize_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)