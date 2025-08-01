#!/usr/bin/env python3
"""
Build and upload script for OmniAvatar package to PyPI.

Usage:
    python scripts/build_and_upload.py --test     # Upload to TestPyPI
    python scripts/build_and_upload.py --prod     # Upload to PyPI
    python scripts/build_and_upload.py --build    # Only build the package
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

def run_command(cmd, description=""):
    """Run a shell command and handle errors."""
    print(f"Running: {cmd}")
    if description:
        print(f"Description: {description}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {description or cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    else:
        print(f"Success: {description or cmd}")
        if result.stdout:
            print(f"Output: {result.stdout}")
    
    return result

def clean_build_artifacts():
    """Clean previous build artifacts."""
    dirs_to_clean = ['build', 'dist', 'OmniAvatar.egg-info']
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            print(f"Cleaning {dir_name}/")
            shutil.rmtree(dir_name)
    
    # Clean __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs[:]:
            if dir_name == '__pycache__':
                print(f"Cleaning {os.path.join(root, dir_name)}/")
                shutil.rmtree(os.path.join(root, dir_name))
                dirs.remove(dir_name)

def check_required_files():
    """Check if all required files exist."""
    required_files = [
        'pyproject.toml',
        'README.md',
        'LICENSE.txt',
        'OmniAvatar/__init__.py',
        'MANIFEST.in'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        sys.exit(1)
    
    print("All required files found.")

def install_build_dependencies():
    """Install required build dependencies."""
    dependencies = [
        'build',
        'twine',
        'setuptools>=45',
        'wheel',
        'setuptools_scm>=6.2'
    ]
    
    for dep in dependencies:
        run_command(f"pip install {dep}", f"Installing {dep}")

def build_package():
    """Build the package."""
    print("Building package...")
    run_command("python -m build", "Building wheel and source distribution")
    
    # List built files
    if os.path.exists('dist'):
        print("\nBuilt files:")
        for file in os.listdir('dist'):
            file_path = os.path.join('dist', file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {file} ({file_size:.2f} MB)")

def validate_package():
    """Validate the built package."""
    print("Validating package...")
    run_command("twine check dist/*", "Validating package with twine")

def upload_to_testpypi():
    """Upload package to TestPyPI."""
    print("Uploading to TestPyPI...")
    print("Make sure you have set TWINE_USERNAME and TWINE_PASSWORD environment variables")
    print("Or use: python -m twine upload --repository testpypi dist/* -u __token__ -p <your-token>")
    
    run_command(
        "python -m twine upload --repository testpypi dist/*",
        "Uploading to TestPyPI"
    )
    
    print("\nTest installation with:")
    print("pip install --index-url https://test.pypi.org/simple/ omniavatar")

def upload_to_pypi():
    """Upload package to PyPI."""
    print("Uploading to PyPI...")
    print("Make sure you have set TWINE_USERNAME and TWINE_PASSWORD environment variables")
    print("Or use: python -m twine upload dist/* -u __token__ -p <your-token>")
    
    # Confirm before uploading to production
    confirm = input("Are you sure you want to upload to PyPI? (yes/no): ")
    if confirm.lower() != 'yes':
        print("Upload cancelled.")
        return
    
    run_command("python -m twine upload dist/*", "Uploading to PyPI")
    
    print("\nPackage uploaded! Install with:")
    print("pip install omniavatar")

def main():
    parser = argparse.ArgumentParser(description="Build and upload OmniAvatar package")
    parser.add_argument('--build', action='store_true', help='Only build the package')
    parser.add_argument('--test', action='store_true', help='Upload to TestPyPI')
    parser.add_argument('--prod', action='store_true', help='Upload to PyPI')
    parser.add_argument('--clean', action='store_true', help='Clean build artifacts')
    
    args = parser.parse_args()
    
    if not any([args.build, args.test, args.prod, args.clean]):
        parser.print_help()
        sys.exit(1)
    
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    if args.clean:
        clean_build_artifacts()
        return
    
    # Check requirements
    check_required_files()
    
    # Clean previous builds
    clean_build_artifacts()
    
    # Install build dependencies
    install_build_dependencies()
    
    # Build package
    build_package()
    
    # Validate package
    validate_package()
    
    if args.test:
        upload_to_testpypi()
    elif args.prod:
        upload_to_pypi()
    
    print("\nDone!")

if __name__ == "__main__":
    main()