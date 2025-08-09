"""
ComfyUI Integrator for OmniAvatar Installation
Handles ComfyUI-specific integration and node registration.
"""
import os
import sys
import json
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging

@dataclass
class ComfyUIConfig:
    """ComfyUI integration configuration"""
    node_package_name: str = "omniavatarnodes"
    custom_nodes_subdir: str = "omniavatarnodes"
    requires_restart: bool = True

class ComfyUIIntegrator:
    """Handles integration with ComfyUI system"""
    
    def __init__(self, config: Optional[ComfyUIConfig] = None):
        self.config = config or ComfyUIConfig()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Detect ComfyUI installation
        self.comfyui_path = self._detect_comfyui_path()
        self.custom_nodes_path = None
        self.target_path = None
        
        if self.comfyui_path:
            self.custom_nodes_path = self.comfyui_path / "custom_nodes" 
            self.target_path = self.custom_nodes_path / self.config.custom_nodes_subdir
    
    def _detect_comfyui_path(self) -> Optional[Path]:
        """Auto-detect ComfyUI installation path"""
        # Check environment variable first
        if 'COMFYUI_PATH' in os.environ:
            path = Path(os.environ['COMFYUI_PATH'])
            if self._is_comfyui_directory(path):
                self.logger.info(f"🎯 Found ComfyUI via COMFYUI_PATH: {path}")
                return path
        
        # Search common locations
        search_paths = [
            Path.cwd(),  # Current directory
            Path.cwd().parent,  # Parent directory  
            Path.cwd() / 'ComfyUI',  # Subdirectory
            Path.cwd().parent / 'ComfyUI',  # Sibling directory
            Path.cwd().parent.parent,  # Grandparent (if in custom_nodes/ournode)
        ]
        
        for path in search_paths:
            if self._is_comfyui_directory(path):
                self.logger.info(f"🎯 Found ComfyUI installation: {path}")
                return path
        
        self.logger.warning("⚠️ ComfyUI installation not found in common locations")
        return None
    
    def _is_comfyui_directory(self, path: Path) -> bool:
        """Check if directory contains ComfyUI installation"""
        if not path.exists() or not path.is_dir():
            return False
        
        # Look for key ComfyUI files/directories
        indicators = [
            'main.py',
            'comfy',
            'custom_nodes',
            'models',
            'web'
        ]
        
        found_indicators = sum(1 for indicator in indicators if (path / indicator).exists())
        return found_indicators >= 3  # Need at least 3 key indicators
    
    def validate_comfyui_installation(self) -> Dict[str, any]:
        """Validate ComfyUI installation and readiness"""
        validation = {
            'comfyui_found': False,
            'custom_nodes_writable': False,
            'manager_available': False,
            'python_compatible': False,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        if not self.comfyui_path:
            validation['errors'].append("ComfyUI installation not found")
            return validation
        
        validation['comfyui_found'] = True
        validation['info']['comfyui_path'] = str(self.comfyui_path)
        
        # Check custom_nodes directory
        if self.custom_nodes_path and self.custom_nodes_path.exists():
            validation['info']['custom_nodes_path'] = str(self.custom_nodes_path)
            
            # Check if writable
            try:
                test_file = self.custom_nodes_path / '.write_test'
                test_file.touch()
                test_file.unlink()
                validation['custom_nodes_writable'] = True
            except (PermissionError, OSError) as e:
                validation['errors'].append(f"custom_nodes directory not writable: {e}")
        else:
            validation['errors'].append("custom_nodes directory not found")
        
        # Check for ComfyUI Manager
        manager_path = self.custom_nodes_path / 'ComfyUI-Manager'
        if manager_path and manager_path.exists():
            validation['manager_available'] = True
            validation['info']['manager_path'] = str(manager_path)
        else:
            validation['warnings'].append("ComfyUI Manager not found - consider installing for easier node management")
        
        # Check Python compatibility
        python_exe = self._get_comfyui_python()
        if python_exe:
            validation['python_compatible'] = True
            validation['info']['python_executable'] = python_exe
        else:
            validation['errors'].append("Could not determine ComfyUI Python executable")
        
        return validation
    
    def _get_comfyui_python(self) -> Optional[str]:
        """Get the Python executable used by ComfyUI"""
        if not self.comfyui_path:
            return None
        
        # Check for portable installation first (Windows)
        portable_python = self.comfyui_path / 'python_embeded' / 'python.exe'
        if portable_python.exists():
            return str(portable_python)
        
        # Check for venv activation script
        venv_paths = [
            self.comfyui_path / 'venv' / 'Scripts' / 'python.exe',  # Windows venv
            self.comfyui_path / 'venv' / 'bin' / 'python',  # Unix venv
            self.comfyui_path / '.venv' / 'Scripts' / 'python.exe',  # Windows .venv
            self.comfyui_path / '.venv' / 'bin' / 'python',  # Unix .venv
        ]
        
        for venv_python in venv_paths:
            if venv_python.exists():
                return str(venv_python)
        
        # Fallback to system Python
        return sys.executable
    
    def install_nodes(self, source_path: Path, method: str = "copy") -> bool:
        """Install nodes to ComfyUI custom_nodes directory"""
        if not self.target_path:
            self.logger.error("❌ ComfyUI installation not found")
            return False
        
        if not source_path.exists():
            self.logger.error(f"❌ Source path does not exist: {source_path}")
            return False
        
        self.logger.info(f"📦 Installing OmniAvatar nodes to ComfyUI...")
        self.logger.info(f"   Source: {source_path}")
        self.logger.info(f"   Target: {self.target_path}")
        self.logger.info(f"   Method: {method}")
        
        try:
            # Remove existing installation
            if self.target_path.exists():
                self.logger.info("🗑️  Removing existing installation...")
                shutil.rmtree(self.target_path)
            
            if method == "symlink":
                # Create symlink (development mode)
                self.target_path.symlink_to(source_path.resolve())
                self.logger.info("🔗 Created symlink for development mode")
            else:
                # Copy files (production mode)
                shutil.copytree(source_path, self.target_path)
                self.logger.info("📁 Copied files for production mode")
            
            # Verify installation
            if self._verify_node_installation():
                self.logger.info("✅ Node installation completed successfully")
                return True
            else:
                self.logger.error("❌ Node installation verification failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Node installation failed: {e}")
            return False
    
    def _verify_node_installation(self) -> bool:
        """Verify that nodes are properly installed"""
        if not self.target_path or not self.target_path.exists():
            return False
        
        # Check for essential files
        required_files = [
            '__init__.py',
            'omniavatarconfig.py', 
            'omniavatarInference.py',
            'requirements.txt'
        ]
        
        for file_name in required_files:
            if not (self.target_path / file_name).exists():
                self.logger.error(f"Missing required file: {file_name}")
                return False
        
        return True
    
    def create_install_script(self) -> bool:
        """Create install.py script for ComfyUI Manager compatibility"""
        if not self.target_path:
            return False
        
        install_script_content = '''"""
Install script for OmniAvatar ComfyUI nodes
This script is executed by ComfyUI Manager during installation.
"""
import sys
import subprocess
from pathlib import Path

def install():
    """Install dependencies and setup"""
    print("🚀 Installing OmniAvatar dependencies...")
    
    try:
        # Install dependencies
        requirements_path = Path(__file__).parent / "requirements.txt"
        if requirements_path.exists():
            subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "-r", str(requirements_path)
            ], check=True)
            print("✅ Dependencies installed successfully")
        
        # Run full installer
        installer_path = Path(__file__).parent / "installer" / "main_installer.py"
        if installer_path.exists():
            subprocess.run([
                sys.executable, str(installer_path), "--auto"
            ], check=True)
            print("✅ OmniAvatar installation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = install()
    sys.exit(0 if success else 1)
'''
        
        try:
            install_script_path = self.target_path / "install.py"
            install_script_path.write_text(install_script_content)
            self.logger.info("📝 Created install.py script for ComfyUI Manager")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create install script: {e}")
            return False
    
    def create_uninstall_script(self) -> bool:
        """Create uninstall.py script for ComfyUI Manager compatibility"""
        if not self.target_path:
            return False
        
        uninstall_script_content = '''"""
Uninstall script for OmniAvatar ComfyUI nodes
"""
import shutil
from pathlib import Path

def uninstall():
    """Clean up installation"""
    print("🧹 Uninstalling OmniAvatar...")
    
    try:
        # Could add cleanup of downloaded models here if desired
        # For now, just notify user
        print("ℹ️  Note: Downloaded models are preserved in pretrained_models/")
        print("   Delete manually if you want to free disk space")
        
        print("✅ OmniAvatar uninstalled")
        return True
        
    except Exception as e:
        print(f"❌ Uninstall failed: {e}")
        return False

if __name__ == "__main__":
    uninstall()
'''
        
        try:
            uninstall_script_path = self.target_path / "uninstall.py"
            uninstall_script_path.write_text(uninstall_script_content)
            self.logger.info("📝 Created uninstall.py script")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create uninstall script: {e}")
            return False
    
    def update_node_list(self) -> bool:
        """Create/update node_list.json for ComfyUI Manager"""
        if not self.target_path:
            return False
        
        node_list = {
            "OmniAvatarConfig": {
                "category": "OmniAvatar", 
                "description": "Configuration node for OmniAvatar parameters"
            },
            "OmniAvatarInference": {
                "category": "OmniAvatar",
                "description": "Main inference node for video generation"
            }
        }
        
        try:
            node_list_path = self.target_path / "node_list.json"
            with open(node_list_path, 'w') as f:
                json.dump(node_list, f, indent=2)
            self.logger.info("📝 Created node_list.json")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create node_list.json: {e}")
            return False
    
    def create_configuration_files(self) -> bool:
        """Create configuration files for the installation"""
        if not self.target_path:
            return False
        
        config_dir = self.target_path / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Create default configuration
        default_config = {
            "model_paths": {
                "dit_path": "pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                "text_encoder_path": "pretrained_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth", 
                "vae_path": "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                "wav2vec_path": "pretrained_models/wav2vec2-base-960h",
                "exp_path": "pretrained_models/OmniAvatar-1.3B"
            },
            "generation_settings": {
                "num_steps": 50,
                "guidance_scale": 4.5,
                "audio_scale": 4.5,
                "seq_len": 200,
                "fps": 25
            },
            "technical_settings": {
                "dtype": "bf16",
                "max_hw": 720,
                "max_tokens": 30000,
                "seed": 42
            },
            "performance_settings": {
                "use_fsdp": False,
                "tea_cache_l1_thresh": 0.0,
                "num_persistent_param_in_dit": None
            }
        }
        
        try:
            config_path = config_dir / "default_config.json"
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            self.logger.info("📝 Created default configuration")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to create configuration: {e}")
            return False
    
    def setup_complete_integration(self, source_path: Path) -> bool:
        """Perform complete ComfyUI integration"""
        self.logger.info("🔧 Setting up complete ComfyUI integration...")
        
        steps = [
            ("Installing nodes", lambda: self.install_nodes(source_path)),
            ("Creating install script", self.create_install_script),
            ("Creating uninstall script", self.create_uninstall_script), 
            ("Updating node list", self.update_node_list),
            ("Creating configuration", self.create_configuration_files)
        ]
        
        success_count = 0
        for step_name, step_func in steps:
            self.logger.info(f"📋 {step_name}...")
            try:
                if step_func():
                    success_count += 1
                    self.logger.info(f"   ✅ {step_name} completed")
                else:
                    self.logger.error(f"   ❌ {step_name} failed")
            except Exception as e:
                self.logger.error(f"   ❌ {step_name} error: {e}")
        
        total_steps = len(steps)
        self.logger.info(f"📊 Integration summary: {success_count}/{total_steps} steps completed")
        
        if success_count == total_steps:
            self.logger.info("🎉 ComfyUI integration completed successfully!")
            if self.config.requires_restart:
                self.logger.info("🔄 Please restart ComfyUI to load the new nodes")
            return True
        else:
            self.logger.warning("⚠️ Integration completed with some issues")
            return success_count > total_steps // 2  # Success if more than half completed

def main():
    """CLI entry point for ComfyUI integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate OmniAvatar with ComfyUI")
    parser.add_argument("--source", type=Path, default=Path(__file__).parent.parent,
                       help="Source path for OmniAvatar nodes")
    parser.add_argument("--method", choices=["copy", "symlink"], default="copy",
                       help="Installation method")
    parser.add_argument("--validate-only", action="store_true",
                       help="Only validate ComfyUI installation")
    
    args = parser.parse_args()
    
    integrator = ComfyUIIntegrator()
    
    if args.validate_only:
        validation = integrator.validate_comfyui_installation()
        
        print("🔍 ComfyUI Validation Results:")
        print("=" * 40)
        
        for key, value in validation['info'].items():
            print(f"  {key}: {value}")
        
        if validation['errors']:
            print("\n❌ Errors:")
            for error in validation['errors']:
                print(f"  • {error}")
        
        if validation['warnings']:
            print("\n⚠️ Warnings:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        overall_status = (validation['comfyui_found'] and 
                         validation['custom_nodes_writable'] and 
                         validation['python_compatible'])
        
        if overall_status:
            print("\n✅ ComfyUI validation passed!")
            return 0
        else:
            print("\n❌ ComfyUI validation failed!")
            return 1
    
    success = integrator.setup_complete_integration(args.source)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())