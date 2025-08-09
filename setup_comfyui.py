#!/usr/bin/env python3
"""
OmniAvatar ComfyUI Setup Script
Simple entry point for automated installation.
"""
import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        return False
    return True

def install_psutil():
    """Install psutil if not available (needed for system validation)"""
    try:
        import psutil
        return True
    except ImportError:
        print("📦 Installing psutil for system detection...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Could not install psutil - some system detection may be limited")
            return False

def main():
    """Main setup entry point"""
    print("=" * 60)
    print("🚀 OmniAvatar ComfyUI Automated Setup")
    print("=" * 60)
    print()
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install psutil if needed
    install_psutil()
    
    # Add the comfyui_nodes directory to Python path
    script_dir = Path(__file__).parent
    comfyui_nodes_dir = script_dir / "comfyui_nodes"
    
    if comfyui_nodes_dir.exists():
        sys.path.insert(0, str(comfyui_nodes_dir))
    else:
        print("❌ ComfyUI nodes directory not found!")
        print("   Make sure you're running this from the OmniAvatar root directory")
        return 1
    
    # Import and run installer
    try:
        from installer.main_installer import main as installer_main
        print("🎯 Starting OmniAvatar ComfyUI installation...")
        print()
        return installer_main()
    except ImportError as e:
        print(f"❌ Could not import installer: {e}")
        print("   Make sure all installer files are present in comfyui_nodes/installer/")
        return 1
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Installation cancelled by user")
        exit_code = 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        exit_code = 1
    
    sys.exit(exit_code)