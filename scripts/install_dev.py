#!/usr/bin/env python3
"""
Development installation script for OmniAvatar package.

This script sets up the package for development with editable installation.
"""

import os
import sys
import subprocess
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
        return False
    else:
        print(f"Success: {description or cmd}")
        if result.stdout:
            print(f"Output: {result.stdout}")
    
    return True

def main():
    # Change to project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    print(f"Working directory: {os.getcwd()}")
    
    # Install package in editable mode with development dependencies
    print("Installing OmniAvatar in development mode...")
    
    success = run_command(
        "pip install -e .[dev]",
        "Installing package in editable mode with dev dependencies"
    )
    
    if not success:
        print("Installation failed. Trying without dev dependencies...")
        success = run_command(
            "pip install -e .",
            "Installing package in editable mode"
        )
    
    if success:
        print("\n✅ Development installation completed!")
        print("\nYou can now:")
        print("1. Import OmniAvatar in Python: from OmniAvatar import WanVideoPipeline")
        print("2. Make changes to the code and they will be reflected immediately")
        print("3. Run tests with: pytest (if dev dependencies are installed)")
        
        # Test import
        print("\nTesting import...")
        test_result = run_command(
            'python -c "import OmniAvatar; print(f\'OmniAvatar version: {OmniAvatar.__version__}\')"',
            "Testing package import"
        )
        
        if test_result:
            print("✅ Package import successful!")
        else:
            print("❌ Package import failed. Check dependencies.")
    else:
        print("❌ Installation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()