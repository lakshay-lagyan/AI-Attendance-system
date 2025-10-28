import os
import datetime
import uuid
import base64
import pickle
import traceback
from functools import wraps

import numpy as np
import cv2 as cv
from deepface import DeepFace
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required

# SQLAlchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, LargeBinary, Text, Float, func, text, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.exc import IntegrityError

# Local helper functions (must be implemented in function.py)
from function import (
    generate_camera_stream, get_face_embedding, continuous_learning_update,
    create_3d_template, extract_multi_vector_embeddings, ensemble_matching,
    search_face_faiss, rebuild_faiss_index, initialize_faiss_index, faiss_index
)

# ---------- Flask app ----------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "replace_this_with_a_real_secret")

# ---------- Database setup (PostgreSQL + SQLAlchemy) ----------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:lakshaybazida@localhost:5432/mydb")

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False))
Base = declarative_base()

# ---------- Models ----------
class Admin(Base):
    __tablename__ = "admins"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    profile_image = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, nullable=True, server_default=func.now(), onupdate=datetime.datetime.utcnow)

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(Text, nullable=False)
    department = Column(String(255), default="")
    phone = Column(String(50), default="")
    profile_image = Column(Text, default="")
    status = Column(String(20), default="active", index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, nullable=True, server_default=func.now(), onupdate=datetime.datetime.utcnow)

class Person(Base):
    __tablename__ = "persons"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    embedding = Column(LargeBinary, nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    photos_count = Column(Integer, default=0)
    status = Column(String(20), default="active", index=True)
    template_type = Column(String(50), default="standard")
    avg_quality = Column(Float, default=0.0)
    update_count = Column(Integer, default=0)
    enrollment_date = Column(DateTime, default=datetime.datetime.utcnow)
    last_updated = Column(DateTime, nullable=True, server_default=func.now(), onupdate=datetime.datetime.utcnow)

class Profile(Base):
    __tablename__ = "profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    department = Column(String(255), default="")
    email = Column(String(255), nullable=False, index=True)
    phone = Column(String(50), default="")
    profile_image = Column(Text, default="")
    registered_at = Column(DateTime, default=datetime.datetime.utcnow)

class EnrollmentRequest(Base):
    __tablename__ = "enrollment_requests"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True)
    phone = Column(String(50), default="")
    password_hash = Column(Text, nullable=False)
    images = Column(JSONB, nullable=False)
    status = Column(String(20), default="pending", index=True)
    submitted_at = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    processed_at = Column(DateTime, nullable=True)
    processed_by = Column(String(255), nullable=True)
    rejection_reason = Column(Text, nullable=True)

class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    confidence = Column(Float, default=0.0)
    template_type = Column(String(50), default="standard")
    continuous_learning_active = Column(String(10), default="false")
    method = Column(String(20), default="auto")

# Indexes
Index("ix_profiles_email", Profile.email)
Index("ix_enrollment_requests_email", EnrollmentRequest.email)

# ---------- Create tables (safe for new tables) ----------
Base.metadata.create_all(bind=engine)

# ---------- Schema fixes (idempotent quick-migrations) ----------
def apply_schema_fixes():
    """
    Apply idempotent ALTER TABLE statements for known missing columns.
    Safe to run multiple times.
    """
    sql_statements = [
        "ALTER TABLE admins ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE;",
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP WITHOUT TIME ZONE;",
        "ALTER TABLE persons ADD COLUMN IF NOT EXISTS last_updated TIMESTAMP WITHOUT TIME ZONE;"
    ]

    with engine.begin() as conn:
        for stmt in sql_statements:
            try:
                conn.execute(text(stmt))
            except Exception as e:
                # don't hard-fail startup; log and continue
                print(f"[SchemaFix] Warning running: {stmt} -> {e}")

        # Backfill nulls and set DB default
        try:
            conn.execute(text("UPDATE admins SET updated_at = COALESCE(updated_at, created_at) WHERE updated_at IS NULL;"))
            conn.execute(text("ALTER TABLE admins ALTER COLUMN updated_at SET DEFAULT now();"))
        except Exception:
            pass

        try:
            conn.execute(text("UPDATE users SET updated_at = COALESCE(updated_at, created_at) WHERE updated_at IS NULL;"))
            conn.execute(text("ALTER TABLE users ALTER COLUMN updated_at SET DEFAULT now();"))
        except Exception:
            pass

        try:
            conn.execute(text("UPDATE persons SET last_updated = COALESCE(last_updated, enrollment_date) WHERE last_updated IS NULL;"))
            conn.execute(text("ALTER TABLE persons ALTER COLUMN last_updated SET DEFAULT now();"))
        except Exception:
            pass

    print("[SchemaFix] Completed (idempotent).")

apply_schema_fixes()

# Initialize FAISS (user function) 
try:
    initialize_faiss_index()
except Exception as e:
    print("[FAISS init error]", e)

# Login manager 
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class AdminUser(UserMixin):
    def __init__(self, admin_obj):
        self.id = str(admin_obj.id)
        self.email = admin_obj.email
        self.name = admin_obj.name
        self.role = "admin"
        self.profile_image = admin_obj.profile_image or ""

class RegularUser(UserMixin):
    def __init__(self, user_obj):
        self.id = str(user_obj.id)
        self.email = user_obj.email
        self.name = user_obj.name
        self.role = "user"
        self.department = user_obj.department or ""
        self.profile_image = user_obj.profile_image or ""
        self.status = user_obj.status or "active"

@login_manager.user_loader
def load_user(user_id):
    db = SessionLocal()
    try:
        try:
            uid = uuid.UUID(user_id)
        except Exception:
            return None

        admin = db.query(Admin).filter(Admin.id == uid).first()
        if admin:
            return AdminUser(admin)

        user = db.query(User).filter(User.id == uid).first()
        if user:
            return RegularUser(user)
    except Exception:
        pass
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

# -------------------- Routes --------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        if getattr(current_user, "role", None) == 'admin':
            return redirect(url_for("admin_dashboard"))
        return redirect(url_for("user_dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        if getattr(current_user, "role", None) == 'admin':
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
        admin = db.query(Admin).filter(Admin.email == email).first()
        if admin and check_password_hash(admin.password_hash, password):
            user = AdminUser(admin)
            login_user(user, remember=remember)
            return redirect(url_for("admin_dashboard"))

        user_obj = db.query(User).filter(User.email == email).first()
        if user_obj:
            if user_obj.status == "blocked":
                flash("Your account has been blocked. Contact admin.", "danger")
                return redirect(url_for("login"))

            if check_password_hash(user_obj.password_hash, password):
                user = RegularUser(user_obj)
                login_user(user, remember=remember)
                return redirect(url_for("user_dashboard"))

        flash("Invalid credentials.", "danger")
        return redirect(url_for("login"))
    finally:
        db.close()

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
        existing = db.query(Admin).filter(Admin.email == email).first()
        if existing:
            return jsonify({"status": "failed", "msg": "Admin already exists"}), 400

        new_admin = Admin(
            name=name,
            email=email,
            password_hash=generate_password_hash(password)
        )
        db.add(new_admin)
        db.commit()
        return jsonify({"status": "success", "msg": "Admin created"}), 201
    except IntegrityError:
        db.rollback()
        return jsonify({"status": "failed", "msg": "Admin already exists"}), 409
    except Exception as e:
        db.rollback()
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

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
    db = SessionLocal()
    try:
        admin = db.query(Admin).filter(Admin.id == uuid.UUID(current_user.id)).first()
        if not admin:
            flash("Admin not found", "danger")
            return redirect(url_for("admin_profile"))

        name = request.form.get("name")
        file = request.files.get("profile_image")

        if name:
            admin.name = name

        if file:
            img_data = base64.b64encode(file.read()).decode('utf-8')
            admin.profile_image = img_data

        admin.updated_at = datetime.datetime.utcnow()
        db.commit()
        flash("Profile updated successfully", "success")
    except Exception as e:
        db.rollback()
        flash("Error updating profile", "danger")
        print("[admin_update_profile error]", e)
    finally:
        db.close()

    return redirect(url_for("admin_profile"))

@app.route("/reg")
@admin_required
def reg():
    return render_template("reg_form.html")

@app.route("/enroll", methods=["POST"])
@admin_required
def enroll():
    db = SessionLocal()
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

        # Check if email exists
        if db.query(User).filter(User.email == email).first():
            return jsonify({"status": "failed", "msg": "Email already exists"}), 400

        print(f"[Enrollment] Processing {len(files)} images for {name}")

        # Extract multi-vector embeddings
        embeddings_data = extract_multi_vector_embeddings(files)

        if len(embeddings_data) < 5:
            return jsonify({
                "status": "failed",
                "msg": f"Only {len(embeddings_data)} valid faces detected. Need at least 5."
            }), 400

        # Create 3D template
        template = create_3d_template(embeddings_data)

        if template is None:
            return jsonify({"status": "failed", "msg": "Failed to create face template"}), 400

        # Get profile image
        profile_image = None
        for i, emb_data in enumerate(embeddings_data[:3]):
            try:
                # emb_data expected to contain 'index' referring to original uploaded file index
                if 'index' in emb_data and isinstance(emb_data['index'], int):
                    file = files[emb_data['index']]
                else:
                    file = files[i]
                file.seek(0)
                file_bytes = file.read()
                img_array = np.frombuffer(file_bytes, np.uint8)
                image = cv.imdecode(img_array, cv.IMREAD_COLOR)

                if image is not None:
                    image = cv.resize(image, (300, 300))
                    _, buffer = cv.imencode('.jpg', image)
                    profile_image = base64.b64encode(buffer).decode('utf-8')
                    break
            except Exception as e:
                print(f"[Profile Image Error] {e}")
                continue

        # Save person
        new_person = Person(
            name=name,
            embedding=pickle.dumps(template),
            embedding_dim=int(template['centroid'].shape[0]) if 'centroid' in template else 0,
            photos_count=len(embeddings_data),
            status="active",
            template_type="3D_multi_vector",
            avg_quality=float(np.mean([x.get('quality', 0.0) for x in embeddings_data])),
            update_count=0
        )
        db.add(new_person)

        # Save profile
        new_profile = Profile(
            name=name,
            department=department,
            email=email,
            phone=phone,
            profile_image=profile_image or ""
        )
        db.add(new_profile)

        # Create user account
        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            department=department,
            phone=phone,
            profile_image=profile_image or "",
            status="active"
        )
        db.add(new_user)

        db.commit()

        # Rebuild FAISS index (user-provided function)
        try:
            rebuild_faiss_index(db)
        except Exception as e:
            print("[FAISS rebuild error]", e)

        print(f"[Enrollment] Successfully enrolled {name}")

        return jsonify({
            "status": "success",
            "name": name,
            "photos_used": len(embeddings_data),
            "embedding_dim": int(template['centroid'].shape[0]) if 'centroid' in template else 0,
            "avg_quality": round(float(np.mean([x.get('quality', 0.0) for x in embeddings_data])), 2),
            "template_type": "3D Multi-Vector"
        })

    except Exception as e:
        db.rollback()
        print("[Enrollment Fatal Error]", e)
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

@app.route("/registration_request")
def registration_request():
    return render_template("registration_req.html")

@app.route("/submit_enrollment_request", methods=["POST"])
def submit_enrollment_request():
    db = SessionLocal()
    try:
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "")
        files = request.files.getlist("files")

        if not name or not email or not password:
            return jsonify({"status": "failed", "msg": "Name, email and password required"}), 400

        if not files or len(files) < 5:
            return jsonify({"status": "failed", "msg": "Please upload at least 5 face images"}), 400

        # Check existing pending request
        existing = db.query(EnrollmentRequest).filter(
            EnrollmentRequest.email == email,
            EnrollmentRequest.status == "pending"
        ).first()
        if existing:
            return jsonify({"status": "failed", "msg": "A pending request already exists"}), 400

        # Store images (base64)
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

        # Save request
        new_request = EnrollmentRequest(
            name=name,
            email=email,
            phone=phone,
            password_hash=generate_password_hash(password),
            images=stored_images,
            status="pending"
        )
        db.add(new_request)
        db.commit()

        return jsonify({
            "status": "success",
            "msg": "Enrollment request submitted successfully",
            "name": name
        })
    except Exception as e:
        db.rollback()
        print("[Request Submission Error]", e)
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": "Unexpected server error"}), 500
    finally:
        db.close()

@app.route("/api/enrollment_requests")
@admin_required
def get_enrollment_requests():
    db = SessionLocal()
    try:
        requests = db.query(EnrollmentRequest).filter(
            EnrollmentRequest.status == "pending"
        ).order_by(EnrollmentRequest.submitted_at.desc()).all()

        out = []
        for r in requests:
            out.append({
                "_id": str(r.id),
                "name": r.name,
                "email": r.email,
                "phone": r.phone,
                "status": r.status,
                "submitted_at": r.submitted_at.isoformat() if r.submitted_at else ""
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/enrollment_request/<request_id>")
@admin_required
def get_enrollment_request_detail(request_id):
    db = SessionLocal()
    try:
        req = db.query(EnrollmentRequest).filter(
            EnrollmentRequest.id == uuid.UUID(request_id)
        ).first()

        if req:
            return jsonify({
                "_id": str(req.id),
                "name": req.name,
                "email": req.email,
                "phone": req.phone,
                "images": req.images,
                "status": req.status,
                "submitted_at": req.submitted_at.isoformat() if req.submitted_at else ""
            })
        return jsonify({"error": "Request not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    finally:
        db.close()

@app.route("/api/approve_enrollment/<request_id>", methods=["POST"])
@admin_required
def approve_enrollment(request_id):
    db = SessionLocal()
    try:
        req = db.query(EnrollmentRequest).filter(
            EnrollmentRequest.id == uuid.UUID(request_id)
        ).first()

        if not req:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404

        if req.status != "pending":
            return jsonify({"status": "failed", "msg": "Request already processed"}), 400

        # Decode images (base64 -> BGR cv images)
        images = []
        for idx, img_b64 in enumerate(req.images):
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
            return jsonify({"status": "failed", "msg": f"Only {len(images)} valid images"}), 400

        embeddings_data = extract_multi_vector_embeddings(images)
        if len(embeddings_data) < 5:
            return jsonify({"status": "failed", "msg": "Not enough valid faces detected"}), 400

        template = create_3d_template(embeddings_data)
        if template is None:
            return jsonify({"status": "failed", "msg": "Failed to create face template"}), 400

        # Choose best quality index if available
        profile_image = None
        try:
            best_idx = embeddings_data[0].get('index', 0)
            img_bytes = base64.b64decode(req.images[best_idx])
            img_array = np.frombuffer(img_bytes, np.uint8)
            image = cv.imdecode(img_array, cv.IMREAD_COLOR)
            if image is not None:
                image = cv.resize(image, (300, 300))
                _, buffer = cv.imencode('.jpg', image)
                profile_image = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[Profile Image Error] {e}")

        department = request.form.get("department", "")

        # Save person
        new_person = Person(
            name=req.name,
            embedding=pickle.dumps(template),
            embedding_dim=int(template['centroid'].shape[0]) if 'centroid' in template else 0,
            photos_count=len(embeddings_data),
            status="active",
            template_type="3D_multi_vector",
            avg_quality=float(np.mean([x.get('quality', 0.0) for x in embeddings_data])),
            update_count=0
        )
        db.add(new_person)

        # Save profile
        new_profile = Profile(
            name=req.name,
            department=department,
            email=req.email,
            phone=req.phone,
            profile_image=profile_image or ""
        )
        db.add(new_profile)

        # Create user account (use existing password hash from request)
        new_user = User(
            name=req.name,
            email=req.email,
            password_hash=req.password_hash,
            department=department,
            phone=req.phone,
            profile_image=profile_image or "",
            status="active"
        )
        db.add(new_user)

        # Update request
        req.status = "approved"
        req.processed_at = datetime.datetime.utcnow()
        req.processed_by = current_user.email if getattr(current_user, "email", None) else None

        db.commit()

        # Rebuild FAISS index
        try:
            rebuild_faiss_index(db)
        except Exception as e:
            print("[FAISS rebuild error]", e)

        return jsonify({
            "status": "success",
            "msg": f"{req.name} has been enrolled successfully",
            "photos_used": len(embeddings_data),
            "avg_quality": round(float(np.mean([x.get('quality', 0.0) for x in embeddings_data])), 2)
        })

    except Exception as e:
        db.rollback()
        print("[Approval Fatal Error]", e)
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

@app.route("/api/reject_enrollment/<request_id>", methods=["POST"])
@admin_required
def reject_enrollment(request_id):
    db = SessionLocal()
    try:
        reason = request.form.get("reason", "Not specified")

        req = db.query(EnrollmentRequest).filter(
            EnrollmentRequest.id == uuid.UUID(request_id)
        ).first()

        if not req:
            return jsonify({"status": "failed", "msg": "Request not found"}), 404

        req.status = "rejected"
        req.rejection_reason = reason
        req.processed_at = datetime.datetime.utcnow()
        req.processed_by = current_user.email if getattr(current_user, "email", None) else None

        db.commit()
        return jsonify({"status": "success", "msg": "Request rejected"})
    except Exception as e:
        db.rollback()
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

@app.route("/api/pending_requests_count")
@admin_required
def pending_requests_count():
    db = SessionLocal()
    try:
        count = db.query(EnrollmentRequest).filter(
            EnrollmentRequest.status == "pending"
        ).count()
        return jsonify({"count": count})
    finally:
        db.close()

@app.route("/api/attendance_recent")
@admin_required
def attendance_recent():
    db = SessionLocal()
    try:
        records = db.query(Attendance).order_by(Attendance.timestamp.desc()).limit(50).all()
        out = []
        for r in records:
            out.append({
                "name": r.name,
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "confidence": r.confidence
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/attendance_stats")
@admin_required
def attendance_stats():
    db = SessionLocal()
    try:
        days = int(request.args.get("days", 30))
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=days - 1)

        records = db.query(Attendance).filter(
            Attendance.timestamp >= start,
            Attendance.timestamp <= end
        ).all()

        counts = {}
        for r in records:
            day = r.timestamp.date().isoformat()
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
        users = db.query(Person).all()
        out = []
        for user in users:
            out.append({
                "_id": str(user.id),
                "name": user.name,
                "photos_count": user.photos_count,
                "enrollment_date": user.enrollment_date.isoformat() if user.enrollment_date else "",
                "status": user.status
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/block_user", methods=["POST"])
@admin_required
def block_user():
    db = SessionLocal()
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"status": "failed", "msg": "Name required"}), 400

        person = db.query(Person).filter(Person.name == name).first()
        if person:
            person.status = "blocked"

        user = db.query(User).filter(User.name == name).first()
        if user:
            user.status = "blocked"

        db.commit()
        try:
            rebuild_faiss_index(db)
        except Exception as e:
            print("[FAISS rebuild error]", e)

        return jsonify({"status": "success", "msg": f"{name} has been blocked"})
    finally:
        db.close()

@app.route("/api/unblock_user", methods=["POST"])
@admin_required
def unblock_user():
    db = SessionLocal()
    try:
        name = request.form.get("name")
        if not name:
            return jsonify({"status": "failed", "msg": "Name required"}), 400

        person = db.query(Person).filter(Person.name == name).first()
        if person:
            person.status = "active"

        user = db.query(User).filter(User.name == name).first()
        if user:
            user.status = "active"

        db.commit()
        try:
            rebuild_faiss_index(db)
        except Exception as e:
            print("[FAISS rebuild error]", e)

        return jsonify({"status": "success", "msg": f"{name} has been unblocked"})
    finally:
        db.close()

@app.route("/user/dashboard")
@login_required
def user_dashboard():
    if getattr(current_user, "role", None) == 'admin':
        return redirect(url_for("admin_dashboard"))
    return render_template("user_dashboard.html", user=current_user)

@app.route("/user/profile")
@login_required
def user_profile():
    db = SessionLocal()
    try:
        profile = db.query(Profile).filter(Profile.email == current_user.email).first()
        return render_template("user_profile.html", profile=profile, user=current_user)
    finally:
        db.close()

@app.route("/user/update_profile", methods=["POST"])
@login_required
def user_update_profile():
    db = SessionLocal()
    try:
        phone = request.form.get("phone")
        file = request.files.get("profile_image")

        img_data = None
        if file:
            img_data = base64.b64encode(file.read()).decode('utf-8')

        # Update profile
        profile = db.query(Profile).filter(Profile.email == current_user.email).first()
        if profile:
            if phone:
                profile.phone = phone
            if img_data:
                profile.profile_image = img_data

        # Update user
        user = db.query(User).filter(User.email == current_user.email).first()
        if user:
            if phone:
                user.phone = phone
            if img_data:
                user.profile_image = img_data

        db.commit()
        flash("Profile updated successfully", "success")
    except Exception as e:
        db.rollback()
        print("[user_update_profile error]", e)
        flash("Error updating profile", "danger")
    finally:
        db.close()

    return redirect(url_for("user_profile"))

@app.route("/api/user/attendance_history")
@login_required
def user_attendance_history():
    db = SessionLocal()
    try:
        records = db.query(Attendance).filter(
            Attendance.name == current_user.name
        ).order_by(Attendance.timestamp.desc()).limit(100).all()

        out = []
        for r in records:
            out.append({
                "timestamp": r.timestamp.isoformat() if r.timestamp else "",
                "confidence": r.confidence
            })
        return jsonify(out)
    finally:
        db.close()

@app.route("/api/user/attendance_stats")
@login_required
def user_attendance_stats():
    db = SessionLocal()
    try:
        days = int(request.args.get("days", 30))
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=days - 1)

        records = db.query(Attendance).filter(
            Attendance.name == current_user.name,
            Attendance.timestamp >= start,
            Attendance.timestamp <= end
        ).all()

        present_days = set()
        for r in records:
            present_days.add(r.timestamp.date().isoformat())

        return jsonify({
            "present_days": len(present_days),
            "total_days": days,
            "percentage": round((len(present_days) / days) * 100, 2) if days > 0 else 0.0
        })
    finally:
        db.close()

@app.route("/api/system_stats")
@admin_required
def system_stats():
    db = SessionLocal()
    try:
        total_users = db.query(Person).count()
        users_with_3d = db.query(Person).filter(Person.template_type == "3D_multi_vector").count()

        avg_updates = db.query(func.avg(Person.update_count)).scalar() or 0
        total_updates = db.query(func.sum(Person.update_count)).scalar() or 0
        avg_quality = db.query(func.avg(Person.avg_quality)).filter(
            Person.avg_quality.isnot(None)
        ).scalar() or 0

        # faiss_index comes from function.py
        size = faiss_index.ntotal if (faiss_index is not None and hasattr(faiss_index, "ntotal")) else 0

        return jsonify({
            "total_users": int(total_users),
            "users_with_3d_template": int(users_with_3d),
            "3d_template_percentage": round((users_with_3d / total_users * 100) if total_users > 0 else 0, 2),
            "avg_updates_per_user": round(float(avg_updates), 2),
            "total_continuous_updates": int(total_updates) if total_updates is not None else 0,
            "avg_enrollment_quality": round(float(avg_quality), 2),
            "faiss_index_size": int(size)
        })
    except Exception as e:
        print(f"[System Stats Error] {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@app.route("/video_feed")
@admin_required
def video_feed():
    return Response(generate_camera_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/mark_attendance", methods=["POST"])
@admin_required
def mark_attendance():
    db = SessionLocal()
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"status": "failed", "msg": "No file uploaded"}), 400

        embedding = get_face_embedding(file)
        if embedding is None:
            return jsonify({"status": "failed", "msg": "No face detected"}), 400

        matched_name, confidence = search_face_faiss(embedding, threshold=0.6)
        if matched_name:
            person = db.query(Person).filter(Person.name == matched_name).first()
            if person and person.status == "blocked":
                return jsonify({"status": "failed", "msg": f"{matched_name} is blocked"}), 403

            template_type = person.template_type if person else "Standard"

            new_attendance = Attendance(
                name=matched_name,
                confidence=confidence,
                template_type=template_type,
                method="manual"
            )
            db.add(new_attendance)
            db.commit()

            return jsonify({
                "status": "success",
                "msg": f"Attendance marked for {matched_name}",
                "name": matched_name,
                "score": round(confidence, 3) if confidence is not None else None,
                "template_type": template_type
            })

        return jsonify({"status": "failed", "msg": "No match found"}), 404
    except Exception as e:
        db.rollback()
        print("[mark_attendance error]", e)
        traceback.print_exc()
        return jsonify({"status": "failed", "msg": str(e)}), 500
    finally:
        db.close()

@app.route("/health")
def health():
    """Health check endpoint for Docker"""
    db = SessionLocal()
    try:
        db.execute(text("SELECT 1"))
        return jsonify({"status": "healthy", "database": "connected"}), 200
    except Exception as e:
        return jsonify({"status": "unhealthy", "error": str(e)}), 500
    finally:
        db.close()

# Cleanup session on teardown
@app.teardown_appcontext
def shutdown_session(exception=None):
    SessionLocal.remove()

if __name__ == "__main__":
    
    app.run( debug=True, port=5000)
