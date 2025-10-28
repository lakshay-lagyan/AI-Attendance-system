
import logging
import os
import base64
import pickle
import datetime
from functools import wraps

import numpy as np
import faiss
import cv2 as cv
from deepface import DeepFace

from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required

from werkzeug.security import generate_password_hash, check_password_hash

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, LargeBinary, Text, Boolean, JSON, Float
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session


from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

# --- Database setup (Postgres + SQLAlchemy) ---
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:password@localhost:5432/mydb")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL, echo=False, future=True, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

# --- Models ---
class AdminModel(Base):
    __tablename__ = "admins"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    profile_image = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class UserModel(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    department = Column(String, default="")
    phone = Column(String, default="")
    profile_image = Column(Text, default="")
    status = Column(String, default="active")  
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class PersonModel(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False) 
    embedding_dim = Column(Integer, nullable=False)
    photos_count = Column(Integer, default=0)
    status = Column(String, default="active")  # active | blocked
    enrollment_date = Column(DateTime, default=datetime.datetime.utcnow)

class ProfileModel(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    department = Column(String, default="")
    email = Column(String, nullable=False, index=True)
    phone = Column(String, default="")
    profile_image = Column(Text, default="")
    registered_at = Column(DateTime, default=datetime.datetime.utcnow)

class EnrollmentRequestModel(Base):
    __tablename__ = "enrollment_requests"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, nullable=False, index=True)
    phone = Column(String, default="")
    password_hash = Column(String, nullable=False)
    images = Column(JSON, nullable=False)  
    status = Column(String, default="pending") 
    submitted_at = Column(DateTime, default=datetime.datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(String, nullable=True)
    rejection_reason = Column(Text, nullable=True)

class AttendanceModel(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    confidence = Column(Float, default=0.0)

# Create tables
Base.metadata.create_all(bind=engine)

# --- FAISS Vector DB Setup ---
EMBEDDING_DIM = 512
faiss_index = None
person_id_map = []

def initialize_faiss_index():
    global faiss_index, person_id_map
    index_path = "/tmp/faiss_index.bin"
    map_path = "/tmp/person_id_map.pkl"
    
    if os.path.exists(index_path) and os.path.exists(map_path):
        try:
            faiss_index = faiss.read_index(index_path)
            with open(map_path, 'rb') as f:
                person_id_map = pickle.load(f)
            logging.info(f"Loaded FAISS index with {faiss_index.ntotal} faces")
        except Exception as e:
            logging.warning(f"Failed to load FAISS index: {e}. Creating new one.")
            faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            person_id_map = []
    else:
        faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        person_id_map = []
        logging.info("Created new FAISS index")

def save_faiss_index():
    if faiss_index is None:
        return
    try:
        faiss.write_index(faiss_index, "/tmp/faiss_index.bin")
        with open("/tmp/person_id_map.pkl", 'wb') as f:
            pickle.dump(person_id_map, f)
    except Exception as e:
        logging.error(f"Failed to save FAISS index: {e}")

def rebuild_faiss_index():
    global faiss_index, person_id_map
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    person_id_map = []
    db = SessionLocal()
    try:
        persons = db.query(PersonModel).filter(PersonModel.status != "blocked").all()
        for p in persons:
            try:
                embedding = pickle.loads(p.embedding)
                if len(embedding) > EMBEDDING_DIM:
                    embedding = embedding[:EMBEDDING_DIM]
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                faiss_index.add(np.array([embedding], dtype=np.float32))
                person_id_map.append(p.name)
            except Exception as e:
                logging.warning("Failed loading embedding for person %s: %s", p.name, e)
        save_faiss_index()
        logging.info("Rebuilt FAISS index with %d persons", len(person_id_map))
    finally:
        db.close()

initialize_faiss_index()

# --- Flask-Login setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class AdminUser(UserMixin):
    def __init__(self, model):
        self.model = model
        self.id = str(model.id)
        self.email = model.email
        self.name = model.name
        self.role = "admin"
        self.profile_image = model.profile_image

class RegularUser(UserMixin):
    def __init__(self, model):
        self.model = model
        self.id = str(model.id)
        self.email = model.email
        self.name = model.name
        self.role = "user"
        self.department = model.department
        self.profile_image = model.profile_image
        self.status = model.status

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        try:
            uid = int(user_id)
        except Exception:
            uid = None

        if uid is not None:
            admin = db.query(AdminModel).filter(AdminModel.id == uid).first()
            if admin:
                return AdminUser(admin)
            user = db.query(UserModel).filter(UserModel.id == uid).first()
            if user:
                return RegularUser(user)
        admin_by_email = db.query(AdminModel).filter(AdminModel.email == str(user_id)).first()
        if admin_by_email:
            return AdminUser(admin_by_email)
        user_by_email = db.query(UserModel).filter(UserModel.email == str(user_id)).first()
        if user_by_email:
            return RegularUser(user_by_email)
    finally:
        db.close()
    return None

def admin_required(f):
    @wraps(f)
    @login_required
    def decorated_function(*args, **kwargs):
        if getattr(current_user, "role", None) != 'admin':
            flash('Access denied. Admin privileges required.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function

# --- Helper functions for embeddings / face recognition ---
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
        logging.exception("Embedding error: %s", e)
        return None

def search_face_faiss(embedding, threshold=0.6):
    if faiss_index is None or faiss_index.ntotal == 0:
        return None, 0.0
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    distances, indices = faiss_index.search(np.array([embedding], dtype=np.float32), k=1)
    if distances[0][0] >= threshold:
        matched_name = person_id_map[indices[0][0]]
        return matched_name, float(distances[0][0])
    return None, 0.0

def is_real_face(face_img):
    try:
        gray = cv.cvtColor(face_img, cv.COLOR_BGR2GRAY)
        blur = cv.Laplacian(gray, cv.CV_64F).var()
        return blur >= 50
    except Exception:
        return False

# --- Routes ---
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
    db = SessionLocal()
    try:
        admin = db.query(AdminModel).filter(AdminModel.email == email).first()
        if admin and check_password_hash(admin.password_hash, password):
            user = AdminUser(admin)
            login_user(user, remember=remember)
            return redirect(url_for("admin_dashboard"))
        user_doc = db.query(UserModel).filter(UserModel.email == email).first()
        if user_doc:
            if user_doc.status == "blocked":
                flash("Your account has been blocked. Contact admin.", "danger")
                return redirect(url_for("login"))
            if check_password_hash(user_doc.password_hash, password):
                user = RegularUser(user_doc)
                login_user(user, remember=remember)
                return redirect(url_for("user_dashboard"))
    finally:
        db.close()
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
    db = SessionLocal()
    try:
        existing = db.query(AdminModel).filter(AdminModel.email == email).first()
        if existing:
            return jsonify({"status": "failed", "msg": "Admin already exists"}), 400
        password_hash = generate_password_hash(password)
        admin = AdminModel(name=name, email=email, password_hash=password_hash, profile_image="", created_at=datetime.datetime.utcnow())
        db.add(admin)
        db.commit()
        return jsonify({"status": "success", "msg": "Admin created"}), 201
    finally:
        db.close()

# Admin pages
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
        db = SessionLocal()
        try:
            admin = db.query(AdminModel).filter(AdminModel.id == int(current_user.id)).first()
            if not admin:
                flash("Admin not found", "danger")
                return redirect(url_for("admin_profile"))
            if name:
                admin.name = name
            if file:
                img_data = base64.b64encode(file.read()).decode('utf-8')
                admin.profile_image = img_data
            db.commit()
            flash("Profile updated successfully", "success")
            return redirect(url_for("admin_profile"))
        finally:
            db.close()
    except Exception as e:
        logging.exception("Error updating admin profile: %s", e)
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
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        files = request.files.getlist("files")
        if not name or not email or not password:
            return jsonify({"status": "failed", "msg": "Name, email and password required"}), 400
        if not files or len(files) < 5:
            return jsonify({"status": "failed", "msg": "Please upload at least 5 face images"}), 400
        db = SessionLocal()
        try:
            if db.query(UserModel).filter(UserModel.email == email).first():
                return jsonify({"status": "failed", "msg": "Email already exists"}), 400
            embeddings = []
            profile_image = None
            for idx, file in enumerate(files, start=1):
                try:
                    file.seek(0)
                    file_bytes = np.frombuffer(file.read(), np.uint8)
                    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
                    if image is None:
                        continue
                    if idx == 1:
                        _, buffer = cv.imencode('.jpg', image)
                        profile_image = base64.b64encode(buffer).decode('utf-8')
                    embedding = get_face_embedding(image)
                    if embedding is not None:
                        embeddings.append(embedding)
                except Exception as e:
                    logging.exception("Enroll file error: %s", e)
                    continue
            if len(embeddings) < 5:
                return jsonify({"status": "failed", "msg": f"Only {len(embeddings)} valid faces found"}), 400
            avg_embedding = np.mean(embeddings, axis=0)
            avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
            # Save Person (embedding pickled)
            person = PersonModel(
                name=name,
                embedding=pickle.dumps(avg_embedding),
                embedding_dim=int(avg_embedding.shape[0]),
                photos_count=len(embeddings),
                status="active",
                enrollment_date=datetime.datetime.utcnow()
            )
            db.add(person)
            # Profile
            profile = ProfileModel(
                name=name,
                department=department,
                email=email,
                phone=phone,
                profile_image=profile_image or "",
                registered_at=datetime.datetime.utcnow()
            )
            db.add(profile)
            # User account
            user = UserModel(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
                department=department,
                phone=phone,
                profile_image=profile_image or "",
                status="active",
                created_at=datetime.datetime.utcnow()
            )
            db.add(user)
            db.commit()
            rebuild_faiss_index()
            return jsonify({"status": "success", "name": name, "photos_used": len(embeddings)})
        finally:
            db.close()
    except Exception as e:
        logging.exception("Enroll fatal error: %s", e)
        return jsonify({"status": "failed", "msg": str(e)}), 500

@app.route("/registration_request")
def registration_request():
    return render_template("registration_req.html")

@app.route("/submit_enrollment_request", methods=["POST"])
def submit_enrollment_request():
    try:
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        files = request.files.getlist("files")
        if not name or not email or not password:
            return jsonify({"status": "failed", "msg": "Name, email and password are required"}), 400
        if not files or len(files) < 5:
            return jsonify({"status": "failed", "msg": "Please upload at least 5 face images"}), 400
        db = SessionLocal()
        try:
            existing = db.query(EnrollmentRequestModel).filter(
                EnrollmentRequestModel.email == email,
                EnrollmentRequestModel.status == "pending"
            ).first()
            if existing:
                return jsonify({"status": "failed", "msg": "A pending request already exists"}), 400
            stored_images = []
            for idx, file in enumerate(files[:10]):
                try:
                    file_bytes = file.read()
                    img_b64 = base64.b64encode(file_bytes).decode('utf-8')
                    stored_images.append(img_b64)
                except Exception as e:
                    logging.exception("Request file error: %s", e)
                    continue
            if len(stored_images) < 5:
                return jsonify({"status": "failed", "msg": "Failed to process images"}), 400
            req = EnrollmentRequestModel(
                name=name,
                email=email,
                phone=phone,
                password_hash=generate_password_hash(password),
                images=stored_images,
                status="pending",
                submitted_at=datetime.datetime.utcnow()
            )
            db.add(req)
            db.commit()
            return jsonify({"status": "success", "msg": "Enrollment request submitted successfully", "name": name})
        finally:
            db.close()
    except Exception as e:
        logging.exception("Request submission error: %s", e)
        return jsonify({"status": "failed", "msg": "Unexpected server error"}), 500

@app.route("/api/enrollment_requests")
@admin_required
def get_enrollment_requests():
    db = SessionLocal()
    try:
        requests = db.query(EnrollmentRequestModel).filter(EnrollmentRequestModel.status == "pending").order_by(EnrollmentRequestModel.submitted_at.desc()).all()
        out = []
        for r in requests:
            out.append({
                "id": r.id,
                "name": r.name,
                "email": r.email,
                "phone": r.phone,
                "status": r.status,
                "submitted_at": r.submitted_at.isoformat() if r.submitted_at else None
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/enrollment_request/<int:request_id>")
@admin_required
def get_enrollment_request_detail(request_id):
    db = SessionLocal()
    try:
        req = db.query(EnrollmentRequestModel).filter(EnrollmentRequestModel.id == request_id).first()
        if not req:
            return jsonify({"error": "Request not found"}), 404
        detail = {
            "id": req.id,
            "name": req.name,
            "email": req.email,
            "phone": req.phone,
            "status": req.status,
            "submitted_at": req.submitted_at.isoformat() if req.submitted_at else None,
            "images": req.images  # base64 images
        }
        return jsonify(detail)
    finally:
        db.close()

@app.route("/api/approve_enrollment/<int:request_id>", methods=["POST"])
@admin_required
def approve_enrollment(request_id):
    db = SessionLocal()
    try:
        req = db.query(EnrollmentRequestModel).filter(EnrollmentRequestModel.id == request_id).first()
        if not req:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404
        if req.status != "pending":
            return jsonify({"status": "failed", "msg": "Request already processed"}), 400
        embeddings = []
        profile_image = None
        for idx, img_b64 in enumerate(req.images):
            try:
                img_bytes = base64.b64decode(img_b64)
                img_array = np.frombuffer(img_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)
                if image is None:
                    continue
                if idx == 0:
                    _, buffer = cv.imencode('.jpg', image)
                    profile_image = base64.b64encode(buffer).decode('utf-8')
                emb = get_face_embedding(image)
                if emb is not None:
                    embeddings.append(emb)
            except Exception as e:
                logging.exception("Approval image process error: %s", e)
                continue
        if len(embeddings) < 5:
            return jsonify({"status": "failed", "msg": f"Only {len(embeddings)} valid faces found"}), 400
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
        department = request.form.get("department", "")
        # Insert into persons, profiles, users
        person = PersonModel(
            name=req.name,
            embedding=pickle.dumps(avg_embedding),
            embedding_dim=int(avg_embedding.shape[0]),
            photos_count=len(embeddings),
            status="active",
            enrollment_date=datetime.datetime.utcnow()
        )
        profile = ProfileModel(
            name=req.name,
            department=department,
            email=req.email,
            phone=req.phone,
            profile_image=profile_image or "",
            registered_at=datetime.datetime.utcnow()
        )
        user = UserModel(
            name=req.name,
            email=req.email,
            password_hash=req.password_hash,  # already hashed earlier
            department=department,
            phone=req.phone,
            profile_image=profile_image or "",
            status="active",
            created_at=datetime.datetime.utcnow()
        )
        db.add(person)
        db.add(profile)
        db.add(user)
        # update enrollment request
        req.status = "approved"
        req.processed_at = datetime.datetime.utcnow()
        req.processed_by = current_user.email
        db.commit()
        rebuild_faiss_index()
        return jsonify({"status": "success", "msg": f"{req.name} has been enrolled successfully", "photos_used": len(embeddings)})
    except Exception as e:
        logging.exception("Approve enrollment fatal error: %s", e)
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

@app.route("/api/reject_enrollment/<int:request_id>", methods=["POST"])
@admin_required
def reject_enrollment(request_id):
    db = SessionLocal()
    try:
        reason = request.form.get("reason", "Not specified")
        req = db.query(EnrollmentRequestModel).filter(EnrollmentRequestModel.id == request_id).first()
        if not req:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404
        req.status = "rejected"
        req.rejection_reason = reason
        req.processed_at = datetime.datetime.utcnow()
        req.processed_by = current_user.email
        db.commit()
        return jsonify({"status": "success", "msg": "Request rejected"})
    except Exception as e:
        logging.exception("Reject enrollment error: %s", e)
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

@app.route("/api/pending_requests_count")
@admin_required
def pending_requests_count():
    db = SessionLocal()
    try:
        count = db.query(EnrollmentRequestModel).filter(EnrollmentRequestModel.status == "pending").count()
        return jsonify({"count": count})
    finally:
        db.close()

@app.route("/api/attendance_recent")
@admin_required
def attendance_recent():
    db = SessionLocal()
    try:
        records = db.query(AttendanceModel).order_by(AttendanceModel.timestamp.desc()).limit(50).all()
        out = []
        for r in records:
            out.append({
                "name": r.name,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "confidence": r.confidence
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/attendance_stats")
@admin_required
def attendance_stats():
    days = int(request.args.get("days", 30))
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days - 1)
    db = SessionLocal()
    try:
        records = db.query(AttendanceModel).filter(AttendanceModel.timestamp >= start, AttendanceModel.timestamp <= end).all()
        counts = {}
        for r in records:
            dt = r.timestamp
            if not dt:
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
    finally:
        db.close()

@app.route("/list_users")
@admin_required
def list_users():
    db = SessionLocal()
    try:
        persons = db.query(PersonModel).all()
        out = []
        for p in persons:
            out.append({
                "id": p.id,
                "name": p.name,
                "photos_count": p.photos_count,
                "enrollment_date": p.enrollment_date.isoformat() if p.enrollment_date else None,
                "status": p.status
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/block_user", methods=["POST"])
@admin_required
def block_user():
    name = request.form.get("name")
    if not name:
        return jsonify({"status": "failed", "msg": "Name required"}), 400
    db = SessionLocal()
    try:
        updated1 = db.query(PersonModel).filter(PersonModel.name == name).update({"status": "blocked"})
        updated2 = db.query(UserModel).filter(UserModel.name == name).update({"status": "blocked"})
        db.commit()
        rebuild_faiss_index()
        return jsonify({"status": "success", "msg": f"{name} has been blocked"})
    finally:
        db.close()

@app.route("/api/unblock_user", methods=["POST"])
@admin_required
def unblock_user():
    name = request.form.get("name")
    if not name:
        return jsonify({"status": "failed", "msg": "Name required"}), 400
    db = SessionLocal()
    try:
        db.query(PersonModel).filter(PersonModel.name == name).update({"status": "active"})
        db.query(UserModel).filter(UserModel.name == name).update({"status": "active"})
        db.commit()
        rebuild_faiss_index()
        return jsonify({"status": "success", "msg": f"{name} has been unblocked"})
    finally:
        db.close()

# User routes
@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if current_user.role == 'admin':
        return redirect(url_for("admin_dashboard"))
    return render_template("user_dashboard.html", user=current_user)

@app.route("/user/profile")
@login_required
def user_profile():
    db = SessionLocal()
    try:
        profile = db.query(ProfileModel).filter(ProfileModel.email == current_user.email).first()
        return render_template("user_profile.html", profile=profile, user=current_user)
    finally:
        db.close()

@app.route("/user/update_profile", methods=["POST"])
@login_required
def user_update_profile():
    try:
        phone = request.form.get("phone")
        file = request.files.get("profile_image")
        db = SessionLocal()
        try:
            profile = db.query(ProfileModel).filter(ProfileModel.email == current_user.email).first()
            user = db.query(UserModel).filter(UserModel.email == current_user.email).first()
            update_needed = False
            if phone:
                if profile:
                    profile.phone = phone
                if user:
                    user.phone = phone
                update_needed = True
            if file:
                img_data = base64.b64encode(file.read()).decode('utf-8')
                if profile:
                    profile.profile_image = img_data
                if user:
                    user.profile_image = img_data
                update_needed = True
            if update_needed:
                db.commit()
            flash("Profile updated successfully", "success")
            return redirect(url_for("user_profile"))
        finally:
            db.close()
    except Exception as e:
        logging.exception("Error updating profile: %s", e)
        flash("Error updating profile", "danger")
        return redirect(url_for("user_profile"))

@app.route("/api/user/attendance_history")
@login_required
def user_attendance_history():
    db = SessionLocal()
    try:
        records = db.query(AttendanceModel).filter(AttendanceModel.name == current_user.name).order_by(AttendanceModel.timestamp.desc()).limit(100).all()
        out = []
        for r in records:
            out.append({
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "confidence": r.confidence
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/user/attendance_stats")
@login_required
def user_attendance_stats():
    days = int(request.args.get("days", 30))
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(days=days - 1)
    db = SessionLocal()
    try:
        records = db.query(AttendanceModel).filter(AttendanceModel.name == current_user.name, AttendanceModel.timestamp >= start, AttendanceModel.timestamp <= end).all()
        present_days = set()
        for r in records:
            if r.timestamp:
                present_days.add(r.timestamp.date().isoformat())
        percentage = round((len(present_days) / days) * 100, 2) if days > 0 else 0.0
        return jsonify({"present_days": len(present_days), "total_days": days, "percentage": percentage})
    finally:
        db.close()

# Camera recognition streaming - DISABLED for Render (no camera access)
def generate_camera_stream():
    """
    WARNING: This function will not work on Render as it requires webcam access.
    Render is a headless server environment without camera hardware.
    Consider removing this route or implementing a file upload alternative.
    """
    try:
        cap = cv.VideoCapture(0)
        threshold = 0.6
        marked_attendance = {}
        cooldown_period = 300
        
        if not cap.isOpened():
            logging.error("Camera not accessible - this is expected on Render")
            # Return a placeholder frame indicating no camera
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv.putText(placeholder, "Camera not available on server", (50, 240), 
                      cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv.imencode(".jpg", placeholder)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            return
        
        while True:
            success, frame = cap.read()
            if not success or frame is None:
                break
            try:
                # Use mtcnn instead of retinaface for TF 2.20 compatibility
                face_objs = DeepFace.extract_faces(img_path=frame, detector_backend="mtcnn", enforce_detection=False, align=False)
                for face_obj in face_objs:
                    facial_area = face_obj["facial_area"]
                    x, y, w, h = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
                    face = frame[y:y+h, x:x+w]
                    if face.size == 0:
                        continue
                    if not is_real_face(face):
                        cv.putText(frame, "Fake Face!", (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        continue
                    embedding = get_face_embedding(face)
                    if embedding is None:
                        continue
                    matched_name, confidence = search_face_faiss(embedding, threshold)
                    if matched_name:
                        # Check if user is blocked
                        db = SessionLocal()
                        try:
                            person = db.query(PersonModel).filter(PersonModel.name == matched_name).first()
                            if person and person.status == "blocked":
                                label = f"{matched_name} - BLOCKED"
                                color = (0, 0, 255)
                                cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                                continue
                        finally:
                            db.close()
                        label = f"{matched_name} ({confidence:.2f})"
                        color = (0, 255, 0)
                        current_time = datetime.datetime.utcnow()
                        last_marked = marked_attendance.get(matched_name)
                        if not last_marked or (current_time - last_marked).total_seconds() > cooldown_period:
                            db = SessionLocal()
                            try:
                                att = AttendanceModel(name=matched_name, timestamp=current_time, confidence=confidence)
                                db.add(att)
                                db.commit()
                                marked_attendance[matched_name] = current_time
                                logging.info("✓ Attendance marked for %s", matched_name)
                            finally:
                                db.close()
                    else:
                        label = "Unknown"
                        color = (0, 165, 255)
                    cv.putText(frame, label, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            except Exception as e:
                logging.exception("Recognition error: %s", e)
            ret, buffer = cv.imencode(".jpg", frame)
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        cap.release()
    except Exception as e:
        logging.exception("Camera stream error: %s", e)

@app.route("/video_feed")
@admin_required
def video_feed():
    """
    WARNING: Video feed will not work on Render (no camera hardware).
    This is kept for local development only.
    """
    return Response(generate_camera_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

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
        db = SessionLocal()
        try:
            person = db.query(PersonModel).filter(PersonModel.name == matched_name).first()
            if person and person.status == "blocked":
                return jsonify({"status": "failed", "msg": f"{matched_name} is blocked"}), 403
            att = AttendanceModel(name=matched_name, timestamp=datetime.datetime.utcnow(), confidence=confidence)
            db.add(att)
            db.commit()
            return jsonify({"status": "success", "msg": f"Attendance marked for {matched_name}", "name": matched_name, "score": round(confidence, 3)})
        finally:
            db.close()
    return jsonify({"status": "failed", "msg": "No match found"}), 404

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()})

# Add all your other routes here (login, logout, admin routes, user routes, etc.)
# [Copy all routes from your original main.py]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)    