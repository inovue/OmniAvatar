"""
Dependency Manager for OmniAvatar ComfyUI Installation
Handles PyTorch and Python package installation.
"""
import os
import sys
import subprocess
import platform
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

@dataclass
class DependencyConfig:
    """Configuration for dependency installation"""
    pytorch_version: str = "2.4.0"
    torchvision_version: str = "0.19.0"
    torchaudio_version: str = "2.4.0"
    cuda_version: str = "cu124"
    pytorch_index_url: str = "https://download.pytorch.org/whl/cu124"
    
    # Optional dependencies
    install_flash_attn: bool = True
    install_xfuser: bool = True

class DependencyManager:
    """Manages installation of Python dependencies"""
    
    def __init__(self, config: Optional[DependencyConfig] = None):
        self.config = config or DependencyConfig()
        self.python_exe = self._get_python_executable()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def _get_python_executable(self) -> str:
        """Get the appropriate Python executable"""
        # Check if we're in a ComfyUI portable installation
        if platform.system() == "Windows":
            # Look for embedded Python in ComfyUI portable
            possible_paths = [
                Path.cwd() / "python_embeded" / "python.exe",
                Path.cwd() / ".." / ".." / "python_embeded" / "python.exe",
                Path(sys.executable)
            ]
        else:
            possible_paths = [Path(sys.executable)]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
        
        return sys.executable
    
    def install_pytorch(self, force_reinstall: bool = False) -> bool:
        """Install PyTorch with CUDA support"""
        try:
            # Check if PyTorch is already installed with correct version
            if not force_reinstall and self._check_pytorch_installed():
                self.logger.info("✅ PyTorch already installed with correct version")
                return True
            
            self.logger.info("🔧 Installing PyTorch with CUDA support...")
            
            # Build PyTorch installation command
            pytorch_packages = [
                f"torch=={self.config.pytorch_version}",
                f"torchvision=={self.config.torchvision_version}",  
                f"torchaudio=={self.config.torchaudio_version}"
            ]
            
            cmd = [
                self.python_exe, "-m", "pip", "install"
            ] + pytorch_packages + [
                "--index-url", self.config.pytorch_index_url,
                "--no-cache-dir"
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ PyTorch installation completed successfully")
                return True
            else:
                self.logger.error(f"❌ PyTorch installation failed:")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ PyTorch installation error: {e}")
            return False
    
    def install_requirements(self, requirements_path: Optional[Path] = None) -> bool:
        """Install packages from requirements.txt"""
        try:
            if requirements_path is None:
                # Default to requirements.txt in project root
                requirements_path = Path(__file__).parent.parent.parent / "requirements.txt"
            
            if not requirements_path.exists():
                self.logger.error(f"❌ Requirements file not found: {requirements_path}")
                return False
            
            self.logger.info(f"🔧 Installing dependencies from {requirements_path}...")
            
            cmd = [
                self.python_exe, "-m", "pip", "install",
                "-r", str(requirements_path),
                "--no-cache-dir"
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Requirements installation completed successfully")
                return True
            else:
                self.logger.error(f"❌ Requirements installation failed:")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Requirements installation error: {e}")
            return False
    
    def install_huggingface_hub(self) -> bool:
        """Install huggingface_hub with CLI support"""
        try:
            self.logger.info("🔧 Installing Hugging Face Hub with CLI support...")
            
            cmd = [
                self.python_exe, "-m", "pip", "install", 
                "huggingface_hub[cli]",
                "--no-cache-dir"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ Hugging Face Hub installation completed")
                return True
            else:
                self.logger.error(f"❌ Hugging Face Hub installation failed:")
                self.logger.error(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Hugging Face Hub installation error: {e}")
            return False
    
    def install_optional_dependencies(self) -> Dict[str, bool]:
        """Install optional dependencies like flash_attn and xfuser"""
        results = {}
        
        if self.config.install_flash_attn:
            results['flash_attn'] = self._install_flash_attn()
        
        if self.config.install_xfuser:
            results['xfuser'] = self._install_xfuser()
        
        return results
    
    def _install_flash_attn(self) -> bool:
        """Install flash_attn for attention acceleration"""
        try:
            self.logger.info("🔧 Installing flash_attn (optional - may take time)...")
            
            cmd = [
                self.python_exe, "-m", "pip", "install", 
                "flash-attn",
                "--no-build-isolation",
                "--no-cache-dir"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
            
            if result.returncode == 0:
                self.logger.info("✅ flash_attn installation completed")
                return True
            else:
                self.logger.warning(f"⚠️ flash_attn installation failed (optional):")
                self.logger.warning(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.warning("⚠️ flash_attn installation timed out (optional)")
            return False
        except Exception as e:
            self.logger.warning(f"⚠️ flash_attn installation error (optional): {e}")
            return False
    
    def _install_xfuser(self) -> bool:
        """Install xfuser for distributed processing"""
        try:
            self.logger.info("🔧 Installing xfuser...")
            
            cmd = [
                self.python_exe, "-m", "pip", "install", 
                "xfuser",
                "--no-cache-dir"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ xfuser installation completed")
                return True
            else:
                self.logger.warning(f"⚠️ xfuser installation failed (optional):")
                self.logger.warning(f"STDERR: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.warning(f"⚠️ xfuser installation error (optional): {e}")
            return False
    
    def _check_pytorch_installed(self) -> bool:
        """Check if PyTorch is installed with correct version and CUDA support"""
        try:
            cmd = [self.python_exe, "-c", 
                   "import torch; print(torch.__version__); print(torch.cuda.is_available())"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    version = lines[0].split('+')[0]  # Remove +cu118 suffix
                    cuda_available = lines[1] == 'True'
                    
                    version_match = version.startswith(self.config.pytorch_version)
                    
                    if version_match and cuda_available:
                        return True
                    else:
                        self.logger.info(f"PyTorch version: {version}, CUDA: {cuda_available}")
                        return False
            
            return False
            
        except Exception:
            return False
    
    def verify_installation(self) -> Dict[str, bool]:
        """Verify that all critical dependencies are working"""
        results = {}
        
        # Check PyTorch
        results['pytorch'] = self._verify_pytorch()
        
        # Check core dependencies
        core_packages = [
            'transformers', 'peft', 'librosa', 'scipy', 'numpy', 'ftfy', 'einops'
        ]
        
        for package in core_packages:
            results[package] = self._verify_package(package)
        
        # Check huggingface_hub
        results['huggingface_hub'] = self._verify_huggingface_hub()
        
        return results
    
    def _verify_pytorch(self) -> bool:
        """Verify PyTorch installation with CUDA"""
        try:
            cmd = [self.python_exe, "-c", """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        print(f"GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
"""]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("✅ PyTorch verification:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        self.logger.info(f"  {line}")
                return True
            else:
                self.logger.error(f"❌ PyTorch verification failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ PyTorch verification error: {e}")
            return False
    
    def _verify_package(self, package_name: str) -> bool:
        """Verify a Python package is importable"""
        try:
            cmd = [self.python_exe, "-c", f"import {package_name}; print('{package_name} OK')"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def _verify_huggingface_hub(self) -> bool:
        """Verify huggingface_hub CLI is available"""
        try:
            cmd = [self.python_exe, "-m", "huggingface_hub", "--help"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    def install_all_dependencies(self) -> bool:
        """Install all required dependencies in correct order"""
        self.logger.info("🚀 Starting comprehensive dependency installation...")
        
        success_count = 0
        total_steps = 4
        
        # Step 1: Install PyTorch
        if self.install_pytorch():
            success_count += 1
        else:
            self.logger.error("❌ Critical: PyTorch installation failed")
            return False
        
        # Step 2: Install core requirements
        if self.install_requirements():
            success_count += 1
        else:
            self.logger.error("❌ Critical: Core requirements installation failed")
            return False
        
        # Step 3: Install HuggingFace Hub
        if self.install_huggingface_hub():
            success_count += 1
        else:
            self.logger.error("❌ Critical: HuggingFace Hub installation failed")
            return False
        
        # Step 4: Install optional dependencies
        optional_results = self.install_optional_dependencies()
        if any(optional_results.values()):  # At least one optional dependency succeeded
            success_count += 1
        
        self.logger.info(f"📊 Dependency installation completed: {success_count}/{total_steps} steps successful")
        
        # Verify installation
        self.logger.info("🔍 Verifying installation...")
        verification_results = self.verify_installation()
        
        failed_packages = [pkg for pkg, success in verification_results.items() if not success]
        if failed_packages:
            self.logger.warning(f"⚠️ Some packages failed verification: {failed_packages}")
        else:
            self.logger.info("✅ All dependencies verified successfully!")
        
        return success_count >= 3  # Core dependencies must succeed

def main():
    """CLI entry point for dependency installation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Install OmniAvatar dependencies")
    parser.add_argument("--force", action="store_true", help="Force reinstall existing packages")
    parser.add_argument("--no-optional", action="store_true", help="Skip optional dependencies")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing installation")
    
    args = parser.parse_args()
    
    config = DependencyConfig()
    if args.no_optional:
        config.install_flash_attn = False
        config.install_xfuser = False
    
    manager = DependencyManager(config)
    
    if args.verify_only:
        results = manager.verify_installation()
        failed = [pkg for pkg, success in results.items() if not success]
        if failed:
            print(f"❌ Verification failed for: {failed}")
            return 1
        else:
            print("✅ All dependencies verified!")
            return 0
    
    success = manager.install_all_dependencies()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())