"""
OmniAvatar ComfyUI Automated Installer Package
"""

from .main_installer import OmniAvatarInstaller, InstallationConfig, InstallationMode
from .system_validator import SystemValidator, ValidationResult
from .dependency_manager import DependencyManager, DependencyConfig
from .model_downloader import ModelDownloader, ModelSize, ModelRegistry
from .comfyui_integrator import ComfyUIIntegrator, ComfyUIConfig

__version__ = "1.0.0"
__author__ = "OmniAvatar Team"
__description__ = "Automated installation system for OmniAvatar ComfyUI nodes"

__all__ = [
    "OmniAvatarInstaller",
    "InstallationConfig", 
    "InstallationMode",
    "SystemValidator",
    "ValidationResult",
    "DependencyManager",
    "DependencyConfig", 
    "ModelDownloader",
    "ModelSize",
    "ModelRegistry",
    "ComfyUIIntegrator",
    "ComfyUIConfig"
]