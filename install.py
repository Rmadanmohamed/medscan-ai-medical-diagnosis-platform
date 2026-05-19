#!/usr/bin/env python3
"""
سكريبت التثبيت السهل لمشروع AI-MedScan Pro
"""

import subprocess
import sys
import os

def run_command(command):
    """تشغيل أمر في سطر الأوامر"""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تشغيل: {command}")
        print(f"رسالة الخطأ: {e.stderr}")
        return False

def main():
    print("🏥 مرحباً بك في تثبيت AI-MedScan Pro")
    print("=" * 50)

    # التحقق من Python
    if sys.version_info < (3, 8):
        print("❌ يتطلب Python 3.8 أو أحدث")
        sys.exit(1)

    print(f"✅ Python {sys.version}")

    # إنشاء البيئة الافتراضية
    print("\n📦 إنشاء البيئة الافتراضية...")
    if not run_command("python -m venv venv"):
        print("❌ فشل في إنشاء البيئة الافتراضية")
        sys.exit(1)

    # تحديد أمر تفعيل البيئة حسب النظام
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Linux/Mac
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"

    print(f"\n📥 تثبيت المتطلبات...")

    # تحديث pip
    if not run_command(f"{pip_cmd} install --upgrade pip"):
        print("⚠️ فشل في تحديث pip")

    # تثبيت المتطلبات
    if not run_command(f"{pip_cmd} install -r requirements.txt"):
        print("❌ فشل في تثبيت المتطلبات")
        sys.exit(1)

    print("\n✅ تم التثبيت بنجاح!")
    print("\n🚀 لتشغيل التطبيق:")
    print(f"1. فعل البيئة الافتراضية: {activate_cmd}")
    print("2. شغل التطبيق: streamlit run ai_medscan_app.py")
    print("\n🌐 سيفتح التطبيق في المتصفح على: http://localhost:8501")

if __name__ == "__main__":
    main()
