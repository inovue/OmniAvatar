"""
Main Installer for OmniAvatar ComfyUI Nodes
Orchestrates the complete automated installation process.
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import logging

# Import our installer components
from .system_validator import SystemValidator
from .dependency_manager import DependencyManager, DependencyConfig
from .model_downloader import ModelDownloader, ModelSize
from .comfyui_integrator import ComfyUIIntegrator

class InstallationMode(Enum):
    """Installation modes"""
    QUICK = "quick"          # 1.3B model only, essential features
    FULL = "full"            # 14B model, all features
    CUSTOM = "custom"        # User selects components

@dataclass
class InstallationConfig:
    """Complete installation configuration"""
    mode: InstallationMode = InstallationMode.QUICK
    model_size: ModelSize = ModelSize.SMALL
    install_optional_deps: bool = True
    force_reinstall: bool = False
    skip_validation: bool = False
    skip_models: bool = False
    skip_dependencies: bool = False
    comfyui_integration_method: str = "copy"
    max_concurrent_downloads: int = 2
    pretrained_models_path: Optional[Path] = None

class InstallationProgress:
    """Track overall installation progress"""
    
    def __init__(self):
        self.phases = [
            "System Validation",
            "Dependency Installation", 
            "Model Download",
            "ComfyUI Integration",
            "Final Verification"
        ]
        self.current_phase = 0
        self.phase_progress = {}
        self.start_time = time.time()
        self.warnings = []
        self.errors = []
    
    def next_phase(self):
        """Move to next installation phase"""
        if self.current_phase < len(self.phases) - 1:
            self.current_phase += 1
    
    def get_current_phase_name(self) -> str:
        """Get name of current phase"""
        return self.phases[self.current_phase] if self.current_phase < len(self.phases) else "Complete"
    
    def get_overall_progress(self) -> float:
        """Get overall progress percentage"""
        base_progress = (self.current_phase / len(self.phases)) * 100
        return min(100.0, base_progress)
    
    def add_warning(self, message: str):
        """Add warning message"""
        self.warnings.append(message)
    
    def add_error(self, message: str):
        """Add error message"""
        self.errors.append(message)

class OmniAvatarInstaller:
    """Main installer orchestrator"""
    
    def __init__(self, config: InstallationConfig):
        self.config = config
        self.progress = InstallationProgress()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.validator = SystemValidator()
        self.dependency_manager = None
        self.model_downloader = None
        self.comfyui_integrator = None
        
    def run_installation(self) -> bool:
        """Run the complete installation process"""
        self.logger.info("🚀 Starting OmniAvatar ComfyUI Installation")
        self.logger.info("=" * 60)
        
        self._print_installation_summary()
        
        # Phase 1: System Validation
        if not self.config.skip_validation:
            self.progress.current_phase = 0
            self._print_phase_header("System Validation")
            if not self._run_system_validation():
                return self._handle_installation_failure("System validation failed")
        
        # Phase 2: Dependency Installation
        if not self.config.skip_dependencies:
            self.progress.next_phase()
            self._print_phase_header("Dependency Installation")
            if not self._run_dependency_installation():
                return self._handle_installation_failure("Dependency installation failed")
        
        # Phase 3: Model Download
        if not self.config.skip_models:
            self.progress.next_phase()
            self._print_phase_header("Model Download")
            if not self._run_model_download():
                return self._handle_installation_failure("Model download failed")
        
        # Phase 4: ComfyUI Integration
        self.progress.next_phase()
        self._print_phase_header("ComfyUI Integration")
        if not self._run_comfyui_integration():
            return self._handle_installation_failure("ComfyUI integration failed")
        
        # Phase 5: Final Verification
        self.progress.next_phase()
        self._print_phase_header("Final Verification")
        if not self._run_final_verification():
            return self._handle_installation_failure("Final verification failed")
        
        # Installation Complete
        return self._handle_installation_success()
    
    def _print_installation_summary(self):
        """Print installation configuration summary"""
        self.logger.info(f"📋 Installation Configuration:")
        self.logger.info(f"   Mode: {self.config.mode.value}")
        self.logger.info(f"   Model Size: {self.config.model_size.value}")
        self.logger.info(f"   Optional Dependencies: {self.config.install_optional_deps}")
        self.logger.info(f"   Integration Method: {self.config.comfyui_integration_method}")
        if self.config.pretrained_models_path:
            self.logger.info(f"   Models Path: {self.config.pretrained_models_path}")
        self.logger.info("")
    
    def _print_phase_header(self, phase_name: str):
        """Print phase header"""
        progress_pct = self.progress.get_overall_progress()
        self.logger.info(f"📍 Phase {self.progress.current_phase + 1}/5: {phase_name} ({progress_pct:.0f}%)")
        self.logger.info("-" * 40)
    
    def _run_system_validation(self) -> bool:
        """Run system validation phase"""
        self.logger.info("🔍 Validating system requirements...")
        
        validation_result = self.validator.validate_system()
        
        if not validation_result.passed:
            self.logger.error("❌ System validation failed!")
            for error in validation_result.errors:
                self.logger.error(f"   • {error}")
                self.progress.add_error(error)
            return False
        
        if validation_result.warnings:
            self.logger.warning("⚠️ System validation warnings:")
            for warning in validation_result.warnings:
                self.logger.warning(f"   • {warning}")
                self.progress.add_warning(warning)
        
        self.logger.info("✅ System validation passed!")
        return True
    
    def _run_dependency_installation(self) -> bool:
        """Run dependency installation phase"""
        self.logger.info("📦 Installing Python dependencies...")
        
        # Configure dependency manager
        dep_config = DependencyConfig()
        dep_config.install_flash_attn = self.config.install_optional_deps
        dep_config.install_xfuser = self.config.install_optional_deps
        
        self.dependency_manager = DependencyManager(dep_config)
        
        success = self.dependency_manager.install_all_dependencies()
        
        if success:
            self.logger.info("✅ Dependency installation completed!")
            return True
        else:
            self.logger.error("❌ Dependency installation failed!")
            self.progress.add_error("Failed to install required dependencies")
            return False
    
    def _run_model_download(self) -> bool:
        """Run model download phase"""
        self.logger.info("⬇️ Downloading OmniAvatar models...")
        
        # Initialize model downloader
        self.model_downloader = ModelDownloader(
            pretrained_models_path=self.config.pretrained_models_path
        )
        
        # Add progress callback
        def progress_callback(model_name: str, progress):
            if hasattr(progress, 'get_progress_percent'):
                pct = progress.get_progress_percent()
                if pct > 0:
                    self.logger.info(f"   📊 {model_name}: {pct:.1f}%")
        
        self.model_downloader.add_progress_callback(progress_callback)
        
        # Download models based on size
        success = self.model_downloader.download_for_model_size(
            self.config.model_size,
            self.config.max_concurrent_downloads
        )
        
        if success:
            self.logger.info("✅ Model download completed!")
            return True
        else:
            self.logger.error("❌ Model download failed!")
            self.progress.add_error("Failed to download required models")
            return False
    
    def _run_comfyui_integration(self) -> bool:
        """Run ComfyUI integration phase"""
        self.logger.info("🔗 Integrating with ComfyUI...")
        
        # Initialize ComfyUI integrator
        self.comfyui_integrator = ComfyUIIntegrator()
        
        # Validate ComfyUI installation
        validation = self.comfyui_integrator.validate_comfyui_installation()
        
        if validation['errors']:
            self.logger.error("❌ ComfyUI integration validation failed:")
            for error in validation['errors']:
                self.logger.error(f"   • {error}")
                self.progress.add_error(error)
            return False
        
        if validation['warnings']:
            for warning in validation['warnings']:
                self.logger.warning(f"⚠️ {warning}")
                self.progress.add_warning(warning)
        
        # Determine source path (current comfyui_nodes directory)
        source_path = Path(__file__).parent.parent
        
        # Setup integration
        success = self.comfyui_integrator.setup_complete_integration(source_path)
        
        if success:
            self.logger.info("✅ ComfyUI integration completed!")
            return True
        else:
            self.logger.error("❌ ComfyUI integration failed!")
            self.progress.add_error("Failed to integrate with ComfyUI")
            return False
    
    def _run_final_verification(self) -> bool:
        """Run final verification phase"""
        self.logger.info("🔍 Running final verification...")
        
        verification_issues = []
        
        # Verify dependencies
        if self.dependency_manager:
            dep_results = self.dependency_manager.verify_installation()
            failed_deps = [pkg for pkg, success in dep_results.items() if not success]
            if failed_deps:
                verification_issues.append(f"Failed dependencies: {failed_deps}")
        
        # Verify models
        if self.model_downloader:
            model_status = self.model_downloader.get_download_status()
            missing_models = [
                info['name'] for key, info in model_status.items() 
                if not info['exists'] and key in self.model_downloader.registry.get_models_for_size(self.config.model_size)
            ]
            if missing_models:
                verification_issues.append(f"Missing models: {missing_models}")
        
        # Verify ComfyUI integration
        if self.comfyui_integrator and self.comfyui_integrator.target_path:
            if not self.comfyui_integrator._verify_node_installation():
                verification_issues.append("ComfyUI node installation verification failed")
        
        if verification_issues:
            self.logger.warning("⚠️ Verification issues found:")
            for issue in verification_issues:
                self.logger.warning(f"   • {issue}")
                self.progress.add_warning(issue)
            return len(verification_issues) <= 2  # Allow minor issues
        
        self.logger.info("✅ Final verification passed!")
        return True
    
    def _handle_installation_failure(self, reason: str) -> bool:
        """Handle installation failure"""
        elapsed = time.time() - self.progress.start_time
        
        self.logger.error("=" * 60)
        self.logger.error(f"❌ Installation FAILED: {reason}")
        self.logger.error(f"⏱️ Time elapsed: {elapsed/60:.1f} minutes")
        
        if self.progress.errors:
            self.logger.error("\n🚨 Critical Errors:")
            for error in self.progress.errors:
                self.logger.error(f"   • {error}")
        
        if self.progress.warnings:
            self.logger.warning("\n⚠️ Warnings:")
            for warning in self.progress.warnings:
                self.logger.warning(f"   • {warning}")
        
        self.logger.error("\n💡 Troubleshooting:")
        self.logger.error("   • Check the error messages above")
        self.logger.error("   • Ensure you have sufficient disk space and VRAM")
        self.logger.error("   • Try running with --force to reinstall components")
        self.logger.error("   • Check your internet connection for model downloads")
        
        return False
    
    def _handle_installation_success(self) -> bool:
        """Handle successful installation"""
        elapsed = time.time() - self.progress.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("🎉 Installation COMPLETED Successfully!")
        self.logger.info(f"⏱️ Total time: {elapsed/60:.1f} minutes")
        
        if self.progress.warnings:
            self.logger.warning(f"\n⚠️ Warnings ({len(self.progress.warnings)}):")
            for warning in self.progress.warnings:
                self.logger.warning(f"   • {warning}")
        
        self.logger.info("\n🎯 Next Steps:")
        self.logger.info("   1. Restart ComfyUI to load the new nodes")
        self.logger.info("   2. Look for 'OmniAvatar' category in the node menu")
        self.logger.info("   3. Start with OmniAvatarConfig node for settings")
        self.logger.info("   4. Connect to OmniAvatarInference for generation")
        
        self.logger.info("\n📚 Usage:")
        self.logger.info("   • Text-to-Video: Set prompt only")
        self.logger.info("   • Image-to-Video: Enable i2v and connect image")
        self.logger.info("   • Audio-Driven: Enable use_audio and connect audio")
        
        # Model-specific recommendations
        if self.config.model_size == ModelSize.SMALL:
            self.logger.info("\n💡 1.3B Model Tips:")
            self.logger.info("   • Faster inference, lower VRAM usage")
            self.logger.info("   • Good for testing and development")
        else:
            self.logger.info("\n💡 14B Model Tips:")
            self.logger.info("   • Higher quality, requires more VRAM")
            self.logger.info("   • Use memory optimization if needed")
        
        return True

def create_installation_config(args) -> InstallationConfig:
    """Create installation config from command line arguments"""
    config = InstallationConfig()
    
    # Set installation mode
    if hasattr(args, 'mode'):
        config.mode = InstallationMode(args.mode)
    
    # Set model size based on mode or explicit argument
    if hasattr(args, 'model_size') and args.model_size:
        config.model_size = ModelSize.SMALL if args.model_size == "1.3B" else ModelSize.LARGE
    elif config.mode == InstallationMode.QUICK:
        config.model_size = ModelSize.SMALL
    elif config.mode == InstallationMode.FULL:
        config.model_size = ModelSize.LARGE
    
    # Set other options
    if hasattr(args, 'no_optional') and args.no_optional:
        config.install_optional_deps = False
    
    if hasattr(args, 'force') and args.force:
        config.force_reinstall = True
    
    if hasattr(args, 'skip_validation') and args.skip_validation:
        config.skip_validation = True
    
    if hasattr(args, 'skip_models') and args.skip_models:
        config.skip_models = True
    
    if hasattr(args, 'skip_dependencies') and args.skip_dependencies:
        config.skip_dependencies = True
    
    if hasattr(args, 'symlink') and args.symlink:
        config.comfyui_integration_method = "symlink"
    
    if hasattr(args, 'models_path') and args.models_path:
        config.pretrained_models_path = Path(args.models_path)
    
    if hasattr(args, 'max_concurrent') and args.max_concurrent:
        config.max_concurrent_downloads = args.max_concurrent
    
    return config

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OmniAvatar ComfyUI Automated Installer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Installation Modes:
  quick    - 1.3B model, essential features (default)
  full     - 14B model, all features  
  custom   - Interactive selection

Examples:
  python main_installer.py                    # Quick installation
  python main_installer.py --mode full        # Full installation
  python main_installer.py --model-size 14B   # Specific model size
  python main_installer.py --force            # Force reinstall
        """
    )
    
    parser.add_argument("--mode", choices=["quick", "full", "custom"], default="quick",
                       help="Installation mode (default: quick)")
    parser.add_argument("--model-size", choices=["1.3B", "14B"], 
                       help="Model size (overrides mode default)")
    parser.add_argument("--no-optional", action="store_true",
                       help="Skip optional dependencies")
    parser.add_argument("--force", action="store_true",
                       help="Force reinstall existing components")
    parser.add_argument("--skip-validation", action="store_true",
                       help="Skip system validation")
    parser.add_argument("--skip-models", action="store_true", 
                       help="Skip model download")
    parser.add_argument("--skip-dependencies", action="store_true",
                       help="Skip dependency installation")
    parser.add_argument("--symlink", action="store_true",
                       help="Use symlinks instead of copying (development mode)")
    parser.add_argument("--models-path", type=str,
                       help="Custom path for pretrained_models directory")
    parser.add_argument("--max-concurrent", type=int, default=2,
                       help="Maximum concurrent downloads")
    parser.add_argument("--auto", action="store_true",
                       help="Run automatically with minimal prompts")
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_installation_config(args)
    
    # Interactive mode selection if custom
    if config.mode == InstallationMode.CUSTOM and not args.auto:
        print("🛠️ Custom Installation Mode")
        print("=" * 30)
        
        # Model size selection
        print("\nSelect model size:")
        print("  1. 1.3B - Faster, lower VRAM (8GB+)")
        print("  2. 14B  - Higher quality, more VRAM (21GB+)")
        choice = input("Choice [1]: ").strip() or "1"
        config.model_size = ModelSize.SMALL if choice == "1" else ModelSize.LARGE
        
        # Optional dependencies
        choice = input("\nInstall optional dependencies (flash_attn, xfuser)? [y/N]: ").strip().lower()
        config.install_optional_deps = choice.startswith('y')
        
        print()
    
    # Run installation
    installer = OmniAvatarInstaller(config)
    success = installer.run_installation()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())