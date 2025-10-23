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

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

# MONGODB
client = MongoClient("mongodb://localhost:27017")
transactional_db = client["transactional_db"]
attendance_col = transactional_db["attendance"]

core = client['secure_db']
persons_col = core["persons"]
profile_col = core["profile"]
admins_col = core["admins"]
users_col = core["users"] 
enrollment_requests_col = core["enrollment_requests"]

doc = {
    "_id": str(datetime.datetime.utcnow().timestamp()) + "_admin@email.com",
    "name": "_Admin",
    "email": "admin@email.com",
    "password_hash": 'password',
    "created_at": datetime.datetime.utcnow()
}

admins_col.insert_one(doc)

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

def rebuild_faiss_index():
    global faiss_index, person_id_map
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    person_id_map = []
    
    for doc in persons_col.find({"status": {"$ne": "blocked"}}):  
        embedding = pickle.loads(doc["embedding"])
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        faiss_index.add(np.array([embedding], dtype=np.float32))
        person_id_map.append(doc["name"])
    
    save_faiss_index()
    print(f"Rebuilt FAISS index with {len(person_id_map)} persons")

initialize_faiss_index()

# Login manager
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

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
    # Try admin first
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
        doc = admins_col.find_one({"_id": ObjectId(user_id)})
        if doc:
            return AdminUser(doc)
        doc = users_col.find_one({"_id": ObjectId(user_id)})
        if doc:
            return RegularUser(doc)
    except:
        pass
    return None

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if current_user.role != 'admin':
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

def get_face_embedding(image):
    try:
        if hasattr(image, 'read'):
            file_bytes = np.frombuffer(image.read(), np.uint8)
            image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
        
        embedding_obj = DeepFace.represent(
            img_path=image,
            model_name='ArcFace',
            detector_backend='opencv',
            enforce_detection=False
        )
        
        if not embedding_obj:
            return None
        
        embedding = np.array(embedding_obj[0]['embedding'])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return None

def search_face_faiss(embedding, threshold=0.6):
    if faiss_index.ntotal == 0:
        return None, 0.0
    
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    distances, indices = faiss_index.search(np.array([embedding], dtype=np.float32), k=1)
    
    if distances[0][0] >= threshold:
        matched_name = person_id_map[indices[0][0]]
        confidence = float(distances[0][0])
        return matched_name, confidence
    return None, 0.0

def is_real_face(face_img):
    try:
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        blur = cv.Laplacian(gray, cv.CV_64F).var()
        return blur >= 50
    except Exception:
        return False

# --- Authentication Routes ---
@app.route("/")
def index():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        if current_user.role == 'admin':
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
    return render_template("reg_form.html")

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
        
        embeddings = []
        profile_image = None
        
        for idx, file in enumerate(files, start=1):
            try:
                file.seek(0)  # Reset file pointer
                file_bytes = np.frombuffer(file.read(), np.uint8)
                image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Use first image as profile
                if idx == 1:
                    _, buffer = cv.imencode('.jpg', image)
                    profile_image = base64.b64encode(buffer).decode('utf-8')
                
                embedding = get_face_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"[Enroll Error] File {idx}: {e}")
                continue
        
        if len(embeddings) < 5:
            return jsonify({"status": "failed", "msg": f"Only {len(embeddings)} valid faces found"}), 400
        
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        # Save to persons collection
        persons_col.insert_one({
            "name": name,
            "embedding": pickle.dumps(avg_embedding),
            "embedding_dim": int(avg_embedding.shape[0]),
            "photos_count": len(embeddings),
            "status": "active",
            "enrollment_date": datetime.datetime.now()
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
        
        rebuild_faiss_index()
        
        return jsonify({
            "status": "success",
            "name": name,
            "photos_used": len(embeddings)
        })
    except Exception as e:
        print("[Enroll Fatal Error]", e)
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
    from bson.objectid import ObjectId
    
    try:
        req = enrollment_requests_col.find_one({"_id": ObjectId(request_id)})
        if not req:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404
        
        if req["status"] != "pending":
            return jsonify({"status": "failed", "msg": "Request already processed"}), 400
        
        embeddings = []
        profile_image = None
        
        for idx, img_b64 in enumerate(req["images"]):
            try:
                img_bytes = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                if idx == 0:
                    _, buffer = cv.imencode('.jpg', image)
                    profile_image = base64.b64encode(buffer).decode('utf-8')
                
                embedding = get_face_embedding(image)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"[Approval Error] Image processing: {e}")
                continue
        
        if len(embeddings) < 5:
            return jsonify({"status": "failed", "msg": f"Only {len(embeddings)} valid faces found"}), 400
        
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        
        department = request.form.get("department", "")
        
        persons_col.insert_one({
            "name": req["name"],
            "embedding": pickle.dumps(avg_embedding),
            "embedding_dim": int(avg_embedding.shape[0]),
            "photos_count": len(embeddings),
            "status": "active",
            "enrollment_date": datetime.datetime.now()
        })
        
        profile_col.insert_one({
            "name": req["name"],
            "department": department,
            "email": req["email"],
            "phone": req["phone"],
            "profile_image": profile_image or "",
            "registered_at": datetime.datetime.now()
        })
        
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
        
        enrollment_requests_col.update_one(
            {"_id": ObjectId(request_id)},
            {"$set": {
                "status": "approved",
                "processed_at": datetime.datetime.now(),
                "processed_by": current_user.email
            }}
        )
        
        rebuild_faiss_index()
        
        return jsonify({
            "status": "success",
            "msg": f"{req['name']} has been enrolled successfully",
            "photos_used": len(embeddings)
        })
    except Exception as e:
        print("[Approval Fatal Error]", e)
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

# --- Camera Recognition ---
def generate_camera_stream():
    cap = cv.VideoCapture(0)
    threshold = 0.6
    marked_attendance = {}
    cooldown_period = 300
    
    if not cap.isOpened():
        print("Camera not accessible")
        return
    
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        
        try:
            face_objs = DeepFace.extract_faces(
                img_path=frame,
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )
            
            for face_obj in face_objs:
                facial_area = face_obj["facial_area"]
                x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                face = frame[y:y+h, x:x+w]
                
                if face.size == 0:
                    continue
                
                if not is_real_face(face):
                    cv.putText(frame, "Fake Face!", (x, y-10),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    continue
                
                embedding = get_face_embedding(face)
                if embedding is None:
                    continue
                
                matched_name, confidence = search_face_faiss(embedding, threshold)
                
                if matched_name:
                    # Check if user is blocked
                    person_doc = persons_col.find_one({"name": matched_name})
                    if person_doc and person_doc.get("status") == "blocked":
                        label = f"{matched_name} - BLOCKED"
                        color = (0, 0, 255)
                        cv.putText(frame, label, (x, y-10),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        continue
                    
                    label = f"{matched_name} ({confidence:.2f})"
                    color = (0, 255, 0)
                    
                    current_time = datetime.datetime.now()
                    last_marked = marked_attendance.get(matched_name)
                    
                    if not last_marked or (current_time - last_marked).total_seconds() > cooldown_period:
                        attendance_col.insert_one({
                            "name": matched_name,
                            "timestamp": current_time.isoformat(),
                            "confidence": confidence
                        })
                        marked_attendance[matched_name] = current_time
                        print(f"✓ Attendance marked for {matched_name}")
                else:
                    label = "Unknown"
                    color = (0, 165, 255)
                
                cv.putText(frame, label, (x, y-10),
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        except Exception as e:
            print(f"[Recognition Error] {e}")
        
        ret, buffer = cv.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    
    cap.release()

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
        
        attendance_col.insert_one({
            "name": matched_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "confidence": confidence
        })
        return jsonify({
            "status": "success",
            "msg": f"Attendance marked for {matched_name}",
            "name": matched_name,
            "score": round(confidence, 3)
        })
    
    return jsonify({"status": "failed", "msg": "No match found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)