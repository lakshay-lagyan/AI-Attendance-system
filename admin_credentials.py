import os
import re
import datetime
import getpass
import uuid
from typing import Optional, List, Dict

from werkzeug.security import generate_password_hash
from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, LargeBinary,
    Float, JSON, Index
)
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker, Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound

# ---------- Configuration ----------
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:lakshaybazida@localhost:5432/mydb"
)

engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

# ---------- Models (inline) ----------
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
    status = Column(String, default="active")  # active | blocked
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

# Optional useful indexes (example)
Index("ix_profiles_email", ProfileModel.email)
Index("ix_enrollment_requests_email", EnrollmentRequestModel.email)

# ---------- Helpers ----------
EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')


def validate_email(email: str) -> bool:
    return bool(email and EMAIL_RE.match(email))


def validate_password(password: str) -> (bool, str):
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"


def ensure_tables_exist():
    Base.metadata.create_all(engine)


def get_session() -> Session:
    return SessionLocal()

# ---------- Core DB operations ----------
def create_admin(name: str, email: str, password: str, profile_image: str = "") -> Dict:
    """
    Create an admin. Returns created admin dict.
    Raises ValueError for invalid input. Raises IntegrityError if email exists.
    """
    if not name or not email or not password:
        raise ValueError("name, email and password are required")

    if not validate_email(email):
        raise ValueError("Invalid email format")

    ok, msg = validate_password(password)
    if not ok:
        raise ValueError(msg)

    password_hash = generate_password_hash(password)

    admin = AdminModel(
        name=name,
        email=email.lower(),
        password_hash=password_hash,
        profile_image=profile_image,
        created_at=datetime.datetime.utcnow()
    )

    session = get_session()
    try:
        session.add(admin)
        session.commit()
        session.refresh(admin)
        return {
            "id": admin.id,
            "name": admin.name,
            "email": admin.email,
            "profile_image": admin.profile_image,
            "created_at": admin.created_at,
        }
    except IntegrityError as ie:
        session.rollback()
        raise
    finally:
        session.close()


def update_admin_password(email: str, new_password: str) -> bool:
    """Update an admin's password. Returns True if updated, False if not found."""
    if not email:
        raise ValueError("email required")
    ok, msg = validate_password(new_password)
    if not ok:
        raise ValueError(msg)

    password_hash = generate_password_hash(new_password)
    session = get_session()
    try:
        admin = session.query(AdminModel).filter(AdminModel.email == email.lower()).one_or_none()
        if not admin:
            return False
        admin.password_hash = password_hash
        session.add(admin)
        session.commit()
        return True
    finally:
        session.close()


def list_admins(limit: int = 100) -> List[Dict]:
    session = get_session()
    try:
        rows = session.query(AdminModel).order_by(AdminModel.created_at.desc()).limit(limit).all()
        return [
            {
                "id": r.id,
                "name": r.name,
                "email": r.email,
                "profile_image": r.profile_image,
                "created_at": r.created_at,
            }
            for r in rows
        ]
    finally:
        session.close()


def get_admin_by_email(email: str) -> Optional[Dict]:
    session = get_session()
    try:
        r = session.query(AdminModel).filter(AdminModel.email == email.lower()).one_or_none()
        if not r:
            return None
        return {
            "id": r.id,
            "name": r.name,
            "email": r.email,
            "profile_image": r.profile_image,
            "created_at": r.created_at,
        }
    finally:
        session.close()

# ---------- CLI helpers ----------
def _save_credentials_file(name: str, email: str, password: str) -> str:
    fname = f"admin_credentials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write("ADMIN CREDENTIALS\n")
        fh.write("=" * 40 + "\n")
        fh.write(f"Name: {name}\n")
        fh.write(f"Email: {email}\n")
        fh.write(f"Password: {password}\n")
        fh.write(f"Created: {datetime.datetime.now()}\n")
        fh.write("=" * 40 + "\n")
        fh.write("\n⚠️ KEEP THIS FILE SECURE!\n")
        fh.write("⚠️ DELETE AFTER SAVING TO PASSWORD MANAGER\n")
    return fname

def create_admin_interactive():
    ensure_tables_exist()
    print("=" * 60)
    print(" " * 15 + "ADMIN ACCOUNT SETUP")
    print("=" * 60)
    print("\nThis will help you create an admin account.\n")

    # name
    while True:
        name = input("Enter admin name: ").strip()
        if name:
            break
        print("❌ Name cannot be empty!\n")

    # email
    while True:
        email = input("Enter admin email: ").strip().lower()
        if not email:
            print("❌ Email cannot be empty!\n")
            continue
        if not validate_email(email):
            print("❌ Invalid email format!\n")
            continue

        existing = get_admin_by_email(email)
        if existing:
            print(f"❌ Admin with email '{email}' already exists!\n")
            choice = input("Do you want to:\n1. Try another email\n2. Update existing admin's password\n3. Exit\nChoice (1/2/3): ").strip()
            if choice == '2':
                update_admin_password_interactive(email)
                return
            elif choice == '3':
                print("Exiting...")
                return
            else:
                continue
        else:
            break

    # password
    print("\n--- CREATE PASSWORD ---")
    print("Password requirements:")
    print("  • Minimum 6 characters")
    print("  • Recommended: Mix of letters, numbers, and symbols\n")

    while True:
        password = getpass.getpass("Enter password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("❌ Passwords don't match!\n")
            continue
        ok, msg = validate_password(password)
        if not ok:
            print(f"❌ {msg}\n")
            continue
        break

    try:
        created = create_admin(name=name, email=email, password=password)
        print("\n" + "=" * 60)
        print("✅ ADMIN ACCOUNT CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Name: {created['name']}")
        print(f"Email: {created['email']}")
        # note: printing plain password for CLI convenience; remove in production
        print(f"Password: {password}")
        print("\n⚠️  IMPORTANT: Save these credentials in a secure location!")
        print("\nYou can now login at: http://localhost:5000/login")
        print("=" * 60)

        save = input("\nDo you want to save credentials to a file? (yes/no): ").strip().lower()
        if save == 'yes':
            fname = _save_credentials_file(name, email, password)
            print(f"\n✅ Credentials saved to: {fname}")
            print("⚠️  Remember to delete this file after saving to a secure location!")
    except IntegrityError:
        print("❌ Email already exists (concurrent insert). Try again.")
    except Exception as e:
        print(f"❌ ERROR: {e}")

def update_admin_password_interactive(email: Optional[str] = None):
    ensure_tables_exist()
    if not email:
        email = input("Enter admin email to update password: ").strip().lower()
    existing = get_admin_by_email(email)
    if not existing:
        print(f"No admin found with email: {email}")
        return

    print(f"\n--- UPDATE PASSWORD for {existing['name']} ({email}) ---")
    while True:
        password = getpass.getpass("Enter new password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("❌ Passwords don't match!\n")
            continue
        ok, msg = validate_password(password)
        if not ok:
            print(f"❌ {msg}\n")
            continue
        break

    try:
        updated = update_admin_password(email, password)
        if updated:
            print("\n✅ PASSWORD UPDATED SUCCESSFULLY!")
            print(f"Email: {email}")
            # don't print the new password in production
        else:
            print("❌ Failed to update - admin not found.")
    except Exception as e:
        print("Error:", e)

def cli_list_admins():
    ensure_tables_exist()
    admins = list_admins()
    if not admins:
        print("\nNo admin accounts found.")
        return
    print("\n" + "=" * 60)
    print("EXISTING ADMIN ACCOUNTS")
    print("=" * 60)
    for idx, a in enumerate(admins, 1):
        print(f"\n{idx}. Name: {a['name']}")
        print(f"   Email: {a['email']}")
        print(f"   Created: {a['created_at']}")
    print("=" * 60)

# ---------- Optional Flask blueprint ----------
try:
    from flask import Blueprint, request, jsonify
    admin_bp = Blueprint("admin_bp", __name__, url_prefix="/admin")

    @admin_bp.route("/admins", methods=["GET"])
    def bp_list_admins():
        try:
            admins = list_admins()
            return jsonify({"status": "ok", "admins": admins})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @admin_bp.route("/admins", methods=["POST"])
    def bp_create_admin():
        data = request.get_json() or {}
        name = data.get("name")
        email = data.get("email")
        password = data.get("password")
        if not (name and email and password):
            return jsonify({"status": "error", "message": "name, email, password required"}), 400
        try:
            created = create_admin(name=name, email=email, password=password, profile_image=data.get("profile_image", ""))
            return jsonify({"status": "ok", "admin": created}), 201
        except IntegrityError:
            return jsonify({"status": "error", "message": "email already exists"}), 409
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @admin_bp.route("/admins/<string:email>/password", methods=["PUT"])
    def bp_update_password(email):
        data = request.get_json() or {}
        new_password = data.get("password")
        if not new_password:
            return jsonify({"status": "error", "message": "password required"}), 400
        try:
            ok = update_admin_password(email=email, new_password=new_password)
            if not ok:
                return jsonify({"status": "error", "message": "admin not found"}), 404
            return jsonify({"status": "ok", "message": "password updated"})
        except ValueError as ve:
            return jsonify({"status": "error", "message": str(ve)}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

except Exception:
    # Flask not installed or blueprint creation failed; keep admin_bp = None
    admin_bp = None

# ---------- CLI entrypoint ----------
def main_menu():
    ensure_tables_exist()
    while True:
        print("\n" + "=" * 60)
        print("SMART ATTENDANCE - ADMIN MANAGEMENT (SQLAlchemy Inline Models)")
        print("=" * 60)
        print("\n1. Create new admin account")
        print("2. Update admin password")
        print("3. List existing admins")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()
        if choice == '1':
            create_admin_interactive()
        elif choice == '2':
            update_admin_password_interactive()
        elif choice == '3':
            cli_list_admins()
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("\n❌ Invalid choice! Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main_menu()
