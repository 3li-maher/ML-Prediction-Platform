#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Prediction Platform
Quick Start Script
Author: ML Development Team
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major < 3 or version.minor < 8:
        print(f"❌ Error: Python 3.8+ required (Found: {version.major}.{version.minor})")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor}.{version.micro} detected")


def check_required_files():
    """Verify all required files exist"""
    required_files = {
        'app_en.py': 'Flask Application',
        'ml_platform_en.html': 'Web Interface',
        'train.csv': 'Titanic Dataset',
        'train1.csv': 'House Prices Dataset'
    }
    
    print("\nChecking required files...")
    missing_files = []
    
    for filename, description in required_files.items():
        if Path(filename).exists():
            print(f"✓ {filename} ({description})")
        else:
            missing_files.append(filename)
            print(f"✗ {filename} ({description}) - MISSING")
    
    if missing_files:
        print(f"\n❌ Error: Missing files: {', '.join(missing_files)}")
        return False
    
    return True


def install_requirements():
    """Install required Python packages"""
    print("\nInstalling required packages...")
    
    packages = [
        'flask==2.3.0',
        'flask-cors==4.0.0',
        'pandas==2.0.0',
        'scikit-learn==1.3.0',
        'numpy==1.24.0',
        'joblib==1.3.0'
    ]
    
    try:
        for package in packages:
            print(f"  Installing {package}...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', package, '-q'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        
        print("✓ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n⚠️  Warning: Installation had issues")
        print("Attempting to run anyway...")
        return True


def open_browser():
    """Open the web browser automatically"""
    time.sleep(3)  # Wait for server to start
    try:
        webbrowser.open('http://localhost:5000')
        print("✓ Browser opened automatically")
    except Exception as e:
        print(f"Note: Could not open browser automatically")
        print("  Please visit: http://localhost:5000")


def start_server():
    """Start the Flask development server"""
    print("\n" + "="*60)
    print("Starting Flask Server...")
    print("="*60)
    
    try:
        # Try to open browser in background thread
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Import and run Flask app
        from app_en import app
        
        print("\n✓ Server is running!")
        print("  URL: http://localhost:5000")
        print("  Press Ctrl+C to stop\n")
        print("="*60)
        
        app.run(debug=True, use_reloader=False, port=5000)
        
    except Exception as e:
        print(f"\n❌ Error starting server: {str(e)}")
        sys.exit(1)


def main():
    """Main execution flow"""
    print("\n" + "="*60)
    print("Machine Learning Prediction Platform")
    print("Quick Start Launcher")
    print("="*60)
    
    # Step 1: Check Python version
    print("\n[1/4] Checking Python version...")
    check_python_version()
    
    # Step 2: Check required files
    print("\n[2/4] Verifying project files...")
    if not check_required_files():
        sys.exit(1)
    
    # Step 3: Install dependencies
    print("\n[3/4] Setting up dependencies...")
    install_requirements()
    
    # Step 4: Start server
    print("\n[4/4] Launching application...")
    try:
        start_server()
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Server stopped by user")
        print("Thank you for using the ML Platform!")
        print("="*60 + "\n")


if __name__ == '__main__':
    main()
