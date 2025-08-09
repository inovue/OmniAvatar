"""
Comprehensive error handling and dependency checking for OmniAvatar ComfyUI nodes.
"""

import sys
import os
import traceback
import warnings
from typing import Dict, List, Any, Optional, Tuple
from functools import wraps
import torch


class DependencyError(Exception):
    """Raised when required dependencies are missing"""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class ModelLoadingError(Exception):
    """Raised when model loading fails"""
    pass


class InferenceError(Exception):
    """Raised when inference fails"""
    pass


def check_dependencies() -> Tuple[bool, List[str]]:
    """
    Check if all required dependencies are available.
    
    Returns:
        Tuple of (all_available, missing_dependencies)
    """
    missing = []
    
    # Core dependencies
    try:
        import torch
        import torchvision
    except ImportError as e:
        missing.append(f"PyTorch/TorchVision: {e}")
    
    try:
        import numpy as np
    except ImportError as e:
        missing.append(f"NumPy: {e}")
    
    try:
        from transformers import Wav2Vec2FeatureExtractor
    except ImportError as e:
        missing.append(f"Transformers: {e}")
    
    try:
        from peft import LoraConfig, inject_adapter_in_model
    except ImportError as e:
        missing.append(f"PEFT (for LoRA): {e}")
    
    # OmniAvatar specific dependencies
    try:
        from OmniAvatar.models.model_manager import ModelManager
        from OmniAvatar.wan_video import WanVideoPipeline
        from OmniAvatar.utils.io_utils import load_state_dict
        from OmniAvatar.models.wav2vec import Wav2VecModel
    except ImportError as e:
        missing.append(f"OmniAvatar package: {e}")
    
    # Audio processing dependencies
    try:
        import librosa
    except ImportError as e:
        missing.append(f"Librosa (for audio processing): {e}")
    
    # Optional but recommended
    try:
        import soundfile
    except ImportError:
        missing.append("SoundFile (optional, for better audio I/O)")
    
    # Distributed processing (optional)
    try:
        import torch.distributed
    except ImportError as e:
        missing.append(f"PyTorch Distributed (optional): {e}")
    
    try:
        from xfuser.core.distributed import initialize_model_parallel
    except ImportError as e:
        missing.append(f"xfuser (optional, for distributed processing): {e}")
    
    return len(missing) == 0, missing


def validate_cuda_setup() -> Tuple[bool, str]:
    """
    Validate CUDA setup for GPU inference.
    
    Returns:
        Tuple of (is_valid, message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA is not available. GPU inference not possible."
    
    try:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_props = torch.cuda.get_device_properties(current_device)
        
        memory_gb = device_props.total_memory / (1024**3)
        if memory_gb < 8:
            return False, f"GPU has only {memory_gb:.1f}GB VRAM. Minimum 8GB recommended for OmniAvatar."
        
        return True, f"CUDA setup valid: {device_count} GPU(s), current device {current_device} with {memory_gb:.1f}GB VRAM"
        
    except Exception as e:
        return False, f"CUDA setup validation failed: {e}"


def validate_model_paths(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that all required model paths exist.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (all_exist, missing_paths)
    """
    missing = []
    
    # Required model paths
    required_paths = {
        'dit_path': 'DiT model',
        'text_encoder_path': 'Text encoder model',
        'vae_path': 'VAE model',
        'exp_path': 'Experiment directory'
    }
    
    for key, description in required_paths.items():
        path = config.get(key, '')
        if not path:
            missing.append(f"{description} path is empty ({key})")
            continue
        
        # Handle comma-separated paths (for dit_path)
        if ',' in path:
            paths = [p.strip() for p in path.split(',')]
            for i, p in enumerate(paths):
                if not os.path.exists(p):
                    missing.append(f"{description} file {i+1} not found: {p}")
        else:
            if not os.path.exists(path):
                missing.append(f"{description} not found: {path}")
    
    # Check for pytorch_model.pt in exp_path
    exp_path = config.get('exp_path', '')
    if exp_path and os.path.exists(exp_path):
        model_file = os.path.join(exp_path, 'pytorch_model.pt')
        if not os.path.exists(model_file):
            missing.append(f"Model file not found: {model_file}")
    
    # Check audio model path if audio is enabled
    if config.get('use_audio', True):
        wav2vec_path = config.get('wav2vec_path', '')
        if not wav2vec_path or not os.path.exists(wav2vec_path):
            missing.append(f"Wav2Vec model required for audio processing: {wav2vec_path}")
    
    return len(missing) == 0, missing


def validate_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required parameters
    required_params = ['dit_path', 'text_encoder_path', 'vae_path', 'exp_path']
    for param in required_params:
        if not config.get(param):
            errors.append(f"Required parameter missing: {param}")
    
    # Validate data types and ranges
    param_validations = {
        'num_steps': (int, 1, 200),
        'guidance_scale': (float, 1.0, 20.0),
        'seq_len': (int, 1, 1000),
        'fps': (int, 1, 60),
        'max_tokens': (int, 1000, 100000),
        'sample_rate': (int, 8000, 48000),
        'sp_size': (int, 1, 8),
    }
    
    for param, (expected_type, min_val, max_val) in param_validations.items():
        value = config.get(param)
        if value is not None:
            if not isinstance(value, expected_type):
                errors.append(f"{param} should be {expected_type.__name__}, got {type(value).__name__}")
            elif not (min_val <= value <= max_val):
                errors.append(f"{param} should be between {min_val} and {max_val}, got {value}")
    
    # Validate enum parameters
    if config.get('dtype') not in ['bf16', 'fp16', 'fp32']:
        errors.append(f"dtype should be one of ['bf16', 'fp16', 'fp32'], got {config.get('dtype')}")
    
    if config.get('max_hw') not in [720, 1280]:
        errors.append(f"max_hw should be 720 or 1280, got {config.get('max_hw')}")
    
    # Validate overlap_frame format
    overlap_frame = config.get('overlap_frame', 13)
    if (overlap_frame - 1) % 4 != 0:
        errors.append(f"overlap_frame must be 1 + 4*n, got {overlap_frame}")
    
    # Validate LoRA parameters if used
    if config.get('train_architecture') == 'lora':
        lora_params = ['lora_rank', 'lora_alpha']
        for param in lora_params:
            value = config.get(param)
            if value is None or not isinstance(value, int) or value < 1:
                errors.append(f"LoRA parameter {param} should be a positive integer, got {value}")
    
    return len(errors) == 0, errors


def safe_execution(func):
    """
    Decorator for safe function execution with comprehensive error handling.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DependencyError as e:
            error_msg = f"Dependency Error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            raise e
        except ConfigurationError as e:
            error_msg = f"Configuration Error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            raise e
        except ModelLoadingError as e:
            error_msg = f"Model Loading Error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            raise e
        except InferenceError as e:
            error_msg = f"Inference Error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            raise e
        except torch.cuda.OutOfMemoryError as e:
            error_msg = f"CUDA Out of Memory in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            print("[SUGGESTION] Try reducing max_tokens, sequence length, or image resolution")
            raise InferenceError(f"GPU out of memory: {e}")
        except FileNotFoundError as e:
            error_msg = f"File Not Found in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            raise ModelLoadingError(f"Required file not found: {e}")
        except Exception as e:
            error_msg = f"Unexpected error in {func.__name__}: {e}"
            print(f"[ERROR] {error_msg}")
            print(f"[TRACEBACK] {traceback.format_exc()}")
            raise RuntimeError(f"Unexpected error: {e}")
    
    return wrapper


def create_error_frame(width: int = 720, height: int = 720, message: str = "Error") -> torch.Tensor:
    """
    Create a simple error frame for display when generation fails.
    
    Args:
        width: Frame width
        height: Frame height  
        message: Error message (not displayed, just for reference)
        
    Returns:
        Error frame tensor in ComfyUI format
    """
    # Create a simple red error frame
    error_frame = torch.zeros((1, height, width, 3), dtype=torch.float32)
    error_frame[..., 0] = 0.8  # Red channel
    return error_frame


def log_system_info():
    """Log system information for debugging."""
    print("[OmniAvatar System Info]")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")


def check_system_requirements() -> Tuple[bool, List[str]]:
    """
    Check overall system requirements for OmniAvatar.
    
    Returns:
        Tuple of (requirements_met, issues)
    """
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        issues.append(f"Python 3.8+ required, got {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch version
    try:
        torch_version = torch.__version__
        major, minor = torch_version.split('.')[:2]
        if int(major) < 2 or (int(major) == 2 and int(minor) < 0):
            issues.append(f"PyTorch 2.0+ recommended, got {torch_version}")
    except Exception as e:
        issues.append(f"Could not verify PyTorch version: {e}")
    
    # Check CUDA
    cuda_valid, cuda_msg = validate_cuda_setup()
    if not cuda_valid:
        issues.append(cuda_msg)
    
    # Check dependencies
    deps_valid, missing_deps = check_dependencies()
    if not deps_valid:
        issues.extend(missing_deps)
    
    return len(issues) == 0, issues


class ErrorReporter:
    """Centralized error reporting and debugging assistance."""
    
    @staticmethod
    def report_dependency_issues():
        """Report dependency installation suggestions."""
        deps_valid, missing = check_dependencies()
        
        if not deps_valid:
            print("\n[OmniAvatar Dependency Issues]")
            print("The following dependencies are missing or have issues:")
            
            for issue in missing:
                print(f"  ❌ {issue}")
            
            print("\n[Installation Suggestions]")
            if any("OmniAvatar" in issue for issue in missing):
                print("  • Install OmniAvatar package from source")
                print("    git clone https://github.com/yourorg/OmniAvatar")
                print("    cd OmniAvatar && pip install -e .")
            
            if any("Transformers" in issue for issue in missing):
                print("  • Install transformers: pip install transformers")
            
            if any("PEFT" in issue for issue in missing):
                print("  • Install PEFT for LoRA support: pip install peft")
            
            if any("Librosa" in issue for issue in missing):
                print("  • Install librosa for audio: pip install librosa")
            
            if any("SoundFile" in issue for issue in missing):
                print("  • Install soundfile for audio I/O: pip install soundfile")
        
        return deps_valid
    
    @staticmethod
    def report_configuration_issues(config: Dict[str, Any]):
        """Report configuration validation issues with suggestions."""
        config_valid, errors = validate_configuration(config)
        paths_valid, path_errors = validate_model_paths(config)
        
        if not config_valid or not paths_valid:
            print("\n[OmniAvatar Configuration Issues]")
            
            all_errors = errors + path_errors
            for error in all_errors:
                print(f"  ❌ {error}")
            
            print("\n[Configuration Suggestions]")
            print("  • Verify all model paths point to downloaded model files")
            print("  • Check that exp_path contains pytorch_model.pt")
            print("  • Ensure parameter values are within valid ranges")
            print("  • For audio generation, verify wav2vec_path is correct")
        
        return config_valid and paths_valid
    
    @staticmethod
    def generate_debug_report(config: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive debug report."""
        report_lines = [
            "=== OmniAvatar Debug Report ===",
            ""
        ]
        
        # System info
        report_lines.extend([
            "System Information:",
            f"  Python: {sys.version}",
            f"  PyTorch: {torch.__version__}",
            f"  CUDA Available: {torch.cuda.is_available()}",
        ])
        
        if torch.cuda.is_available():
            report_lines.extend([
                f"  CUDA Version: {torch.version.cuda}",
                f"  GPU Count: {torch.cuda.device_count()}",
            ])
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report_lines.append(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        
        report_lines.append("")
        
        # Dependencies
        deps_valid, missing = check_dependencies()
        report_lines.extend([
            "Dependencies:",
            f"  Status: {'✓ All Available' if deps_valid else '❌ Missing Dependencies'}",
        ])
        
        if not deps_valid:
            for issue in missing:
                report_lines.append(f"    - {issue}")
        
        report_lines.append("")
        
        # Configuration (if provided)
        if config:
            config_valid, errors = validate_configuration(config)
            paths_valid, path_errors = validate_model_paths(config)
            
            report_lines.extend([
                "Configuration:",
                f"  Config Valid: {'✓ Yes' if config_valid else '❌ No'}",
                f"  Paths Valid: {'✓ Yes' if paths_valid else '❌ No'}",
            ])
            
            all_errors = errors + path_errors
            if all_errors:
                report_lines.append("  Issues:")
                for error in all_errors:
                    report_lines.append(f"    - {error}")
        
        return "\n".join(report_lines)