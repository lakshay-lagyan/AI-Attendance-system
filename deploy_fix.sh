#!/bin/bash

echo "🔧 Fixing deployment issues..."

# Test imports first
echo "📦 Testing imports..."
python3 test_imports.py
if [ $? -ne 0 ]; then
    echo "❌ Import test failed. Please fix dependencies first."
    exit 1
fi

# Check if main_fixed.py exists and has no syntax errors
echo "🔍 Checking main_fixed.py syntax..."
python3 -m py_compile main_fixed.py
if [ $? -ne 0 ]; then
    echo "❌ main_fixed.py has syntax errors"
    exit 1
fi

echo "✅ main_fixed.py syntax is valid"

# Backup current main.py
if [ -f "main.py" ]; then
    echo "📁 Backing up current main.py..."
    cp main.py main_backup_$(date +%Y%m%d_%H%M%S).py
fi

# Replace main.py with the fixed version
echo "🔄 Replacing main.py with fixed version..."
cp main_fixed.py main.py

# Update start.sh to use main.py (now the fixed version)
echo "📝 Updating start.sh..."
sed -i 's/main_fixed:app/main:app/g' start.sh

echo "✅ Deployment fix complete!"
echo ""
echo "Next steps:"
echo "1. Commit and push these changes:"
echo "   git add ."
echo "   git commit -m 'Fix: Replace main.py with syntax-error-free version'"
echo "   git push origin main"
echo ""
echo "2. Your Railway deployment should now work correctly"
echo ""
echo "🔐 Default credentials:"
echo "   Super Admin: superadmin@admin.com / SuperAdmin@123"
echo "   Admin: admin@admin.com / password123"
echo "   ⚠️  CHANGE THESE IMMEDIATELY AFTER DEPLOYMENT!"
