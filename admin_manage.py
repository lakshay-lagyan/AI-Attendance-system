from pymongo import MongoClient
from werkzeug.security import generate_password_hash
import datetime
import getpass
import re

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, "Password is valid"

def create_admin_interactive():
    
    print("=" * 60)
    print(" " * 15 + "ADMIN ACCOUNT SETUP")
    print("=" * 60)
    print("\nThis will help you create an admin account.\n")
    
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient("mongodb://localhost:27017")
        core = client['secure_db']
        admins_col = core["admins"]
        print("✅ Connected to MongoDB\n")
        
        # Get admin name
        while True:
            name = input("Enter admin name: ").strip()
            if name:
                break
            print("❌ Name cannot be empty!\n")
        
        # Get admin email
        while True:
            email = input("Enter admin email: ").strip().lower()
            if not email:
                print("❌ Email cannot be empty!\n")
                continue
            if not validate_email(email):
                print("❌ Invalid email format!\n")
                continue
            
            # Check if email already exists
            existing = admins_col.find_one({"email": email})
            if existing:
                print(f"❌ Admin with email '{email}' already exists!\n")
                choice = input("Do you want to:\n1. Try another email\n2. Update existing admin's password\n3. Exit\nChoice (1/2/3): ")
                
                if choice == '2':
                    # Update password
                    print("\n--- UPDATE PASSWORD ---")
                    while True:
                        password = getpass.getpass("Enter new password: ")
                        confirm = getpass.getpass("Confirm password: ")
                        
                        if password != confirm:
                            print("❌ Passwords don't match!\n")
                            continue
                        
                        valid, msg = validate_password(password)
                        if not valid:
                            print(f"❌ {msg}\n")
                            continue
                        
                        password_hash = generate_password_hash(password)
                        admins_col.update_one(
                            {"email": email},
                            {"$set": {"password_hash": password_hash}}
                        )
                        
                        print("\n" + "=" * 60)
                        print("✅ PASSWORD UPDATED SUCCESSFULLY!")
                        print("=" * 60)
                        print(f"Email: {email}")
                        print(f"New Password: {password}")
                        print("\n⚠️  SAVE THESE CREDENTIALS SECURELY!")
                        print("\nLogin at: http://localhost:5000/login")
                        print("=" * 60)
                        return
                
                elif choice == '3':
                    print("Exiting...")
                    return
                else:
                    continue
            else:
                break
        
        # Get admin password
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
            
            valid, msg = validate_password(password)
            if not valid:
                print(f"❌ {msg}\n")
                continue
            
            break
        
        # Create password hash
        password_hash = generate_password_hash(password)
        
        # Create admin document
        admin_doc = {
            "_id": str(datetime.datetime.utcnow().timestamp()) + "_" + email,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "profile_image": "",
            "created_at": datetime.datetime.utcnow()
        }
        
        # Insert into database
        admins_col.insert_one(admin_doc)
        
        # Success message
        print("\n" + "=" * 60)
        print("✅ ADMIN ACCOUNT CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Name: {name}")
        print(f"Email: {email}")
        print(f"Password: {password}")
        print("\n⚠️  IMPORTANT: Save these credentials in a secure location!")
        print("\nYou can now login at: http://localhost:5000/login")
        print("=" * 60)
        
        # Save to file option
        save = input("\nDo you want to save credentials to a file? (yes/no): ")
        if save.lower() == 'yes':
            filename = f"admin_credentials_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write("ADMIN CREDENTIALS\n")
                f.write("=" * 40 + "\n")
                f.write(f"Name: {name}\n")
                f.write(f"Email: {email}\n")
                f.write(f"Password: {password}\n")
                f.write(f"Created: {datetime.datetime.now()}\n")
                f.write("=" * 40 + "\n")
                f.write("\n⚠️ KEEP THIS FILE SECURE!\n")
                f.write("⚠️ DELETE AFTER SAVING TO PASSWORD MANAGER\n")
            
            print(f"\n✅ Credentials saved to: {filename}")
            print("⚠️  Remember to delete this file after saving to a secure location!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure MongoDB is running: mongod")
        print("2. Check if MongoDB is accessible: mongosh")
        print("3. Verify MongoDB connection string in script")
    finally:
        if 'client' in locals():
            client.close()

# List all existing admin accounts
def list_admins():
    try:
        client = MongoClient("mongodb://localhost:27017")
        core = client['secure_db']
        admins_col = core["admins"]
        
        admins = list(admins_col.find())
        
        if not admins:
            print("\nNo admin accounts found.")
            return
        
        print("\n" + "=" * 60)
        print("EXISTING ADMIN ACCOUNTS")
        print("=" * 60)
        for idx, admin in enumerate(admins, 1):
            print(f"\n{idx}. Name: {admin.get('name')}")
            print(f"   Email: {admin.get('email')}")
            print(f"   Created: {admin.get('created_at')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    while True:
        print("\n" + "=" * 60)
        print("SMART ATTENDANCE - ADMIN MANAGEMENT")
        print("=" * 60)
        print("\n1. Create new admin account")
        print("2. Update admin password")
        print("3. List existing admins")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            create_admin_interactive()
        elif choice == '2':
            print("\nTo update password, choose option 1 and enter the existing email.")
        elif choice == '3':
            list_admins()
        elif choice == '4':
            print("\nGoodbye!")
            break
        else:
            print("\n❌ Invalid choice! Please enter 1, 2, 3, or 4.")