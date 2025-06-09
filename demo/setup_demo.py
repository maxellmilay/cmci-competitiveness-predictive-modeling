#!/usr/bin/env python3
"""
Setup script for the MLflow pipeline demo
"""

import os
import sys
from pathlib import Path

def create_directory_structure():
    """Create necessary directory structure"""
    dirs = [
        "demo/data",
        "demo/config", 
        "demo/logs",
        "artifacts/evaluation",
        "src/data",
        "src/models", 
        "src/evaluation",
        "src/pipeline"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")

def create_init_files():
    """Create __init__.py files for proper imports"""
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/models/__init__.py", 
        "src/evaluation/__init__.py",
        "src/pipeline/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
        print(f"✅ Created: {init_file}")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'mlflow',
        'pandas', 
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pyyaml',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is NOT installed")
    
    if missing_packages:
        print("\n🚨 Missing packages detected!")
        print("Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n🎉 All required packages are installed!")
        return True

def main():
    """Main setup function"""
    print("🔧 Setting up demo environment...")
    print("=" * 40)
    
    # Create directory structure
    print("\n📁 Creating directory structure...")
    create_directory_structure()
    
    # Create __init__.py files
    print("\n📄 Creating __init__.py files...")
    create_init_files()
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n" + "=" * 40)
    if deps_ok:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start MLflow server: mlflow server --host 0.0.0.0 --port 5000")
        print("2. Run demo: python demo/run_pipeline_demo.py")
    else:
        print("❌ Setup incomplete - please install missing packages")
        sys.exit(1)

if __name__ == "__main__":
    main() 