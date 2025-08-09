"""
ComfyUI OmniAvatar Nodes

This package provides ComfyUI integration for OmniAvatar video generation.

Nodes:
- OmniAvatarConfig: Configuration node for all OmniAvatar parameters
- OmniAvatarInference: Main inference node for video generation

Requirements:
- PyTorch 2.0+
- OmniAvatar package
- CUDA GPU with 8GB+ VRAM recommended

Installation:
1. Clone OmniAvatar repository
2. Install dependencies: pip install -r requirements.txt
3. Place these nodes in ComfyUI custom_nodes directory
"""

import warnings
import sys

# Check basic requirements early
try:
    import torch
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available - OmniAvatar will not work without GPU")
except ImportError:
    warnings.warn("PyTorch not found - OmniAvatar nodes require PyTorch")

# Import error handling first
try:
    from .error_handling import check_dependencies, ErrorReporter, log_system_info
    
    # Check dependencies and report issues
    deps_available, missing_deps = check_dependencies()
    if not deps_available:
        print("\n[OmniAvatar ComfyUI Nodes] Dependency issues detected:")
        ErrorReporter.report_dependency_issues()
        print("\nNodes will be registered but may not function properly without these dependencies.\n")
    else:
        print("[OmniAvatar ComfyUI Nodes] All dependencies available ✓")
        
except ImportError as e:
    print(f"[OmniAvatar ComfyUI Nodes] Could not import error handling: {e}")
    deps_available = False

# Import nodes with error handling
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .omniavatarconfig import OmniAvatarConfig
    NODE_CLASS_MAPPINGS["OmniAvatarConfig"] = OmniAvatarConfig
    NODE_DISPLAY_NAME_MAPPINGS["OmniAvatarConfig"] = "OmniAvatar Config"
    print("[OmniAvatar] Config node loaded ✓")
except ImportError as e:
    print(f"[OmniAvatar] Failed to load Config node: {e}")

try:
    from .omniavatarInference import OmniAvatarInference
    NODE_CLASS_MAPPINGS["OmniAvatarInference"] = OmniAvatarInference  
    NODE_DISPLAY_NAME_MAPPINGS["OmniAvatarInference"] = "OmniAvatar Inference"
    print("[OmniAvatar] Inference node loaded ✓")
except ImportError as e:
    print(f"[OmniAvatar] Failed to load Inference node: {e}")

# Web UI integration
WEB_DIRECTORY = "./web"

# Node information for ComfyUI
__version__ = "1.0.0"
__author__ = "OmniAvatar Team"
__description__ = "ComfyUI nodes for OmniAvatar video generation"

def print_startup_info():
    """Print startup information and diagnostics"""
    print("\n" + "="*60)
    print("OmniAvatar ComfyUI Nodes v" + __version__)
    print("="*60)
    
    if deps_available:
        print("✓ All dependencies available")
        print("✓ Ready for video generation")
        
        # Log basic system info
        try:
            log_system_info()
        except Exception as e:
            print(f"Could not log system info: {e}")
    else:
        print("⚠ Dependency issues detected - see above for details")
        print("⚠ Nodes registered but may not function properly")
    
    print(f"✓ Loaded {len(NODE_CLASS_MAPPINGS)} node(s)")
    print("="*60 + "\n")

# Print startup info when module is imported
try:
    print_startup_info()
except Exception as e:
    print(f"[OmniAvatar] Startup info failed: {e}")

__all__ = [
    "NODE_CLASS_MAPPINGS", 
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
    "__version__",
    "__author__",
    "__description__"
]