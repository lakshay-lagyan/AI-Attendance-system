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
from functools import wraps

from function import generate_camera_stream, get_face_embedding, continuous_learning_update, create_3d_template, extract_multi_vector_embeddings, ensemble_matching, search_face_faiss, rebuild_faiss_index

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

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
    return redirect(url_for("login"))

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
    return render_template("admin_dashboard.html", admin=current_user)

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
    name = request.form.get("name")
    if not name:
        return jsonify({"status": "failed", "msg": "Name required"}), 400
    
    persons_col.update_one({"name": name}, {"$set": {"status": "blocked"}})
    users_col.update_one({"name": name}, {"$set": {"status": "blocked"}})
    
    rebuild_faiss_index()
    
    return jsonify({"status": "success", "msg": f"{name} has been blocked"})

@app.route("/api/unblock_user", methods=["POST"])
@admin_required
def unblock_user():
    name = request.form.get("name")
    if not name:
        return jsonify({"status": "failed", "msg": "Name required"}), 400
    
    persons_col.update_one({"name": name}, {"$set": {"status": "active"}})
    users_col.update_one({"name": name}, {"$set": {"status": "active"}})
    
    rebuild_faiss_index()
    
    return jsonify({"status": "success", "msg": f"{name} has been unblocked"})

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

# --- User Routes ---
@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if current_user.role == 'admin':
        return redirect(url_for("admin_dashboard"))
    return render_template("user_dashboard.html", user=current_user)

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