"""
Model Downloader for OmniAvatar ComfyUI Installation
Handles HuggingFace model downloads with progress tracking and resume capability.
"""
import os
import sys
import subprocess
import time
import json
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class ModelSize(Enum):
    """Model size variants"""
    SMALL = "1.3B"
    LARGE = "14B" 

@dataclass
class ModelInfo:
    """Information about a model to download"""
    name: str
    huggingface_repo: str
    local_path: str
    size_gb: float
    required: bool = True
    description: str = ""
    files_to_check: List[str] = field(default_factory=list)

class ModelRegistry:
    """Registry of all OmniAvatar models"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.models = {
            # Base models
            "wan_14b": ModelInfo(
                name="Wan2.1-T2V-14B",
                huggingface_repo="Wan-AI/Wan2.1-T2V-14B",
                local_path=str(base_path / "Wan2.1-T2V-14B"),
                size_gb=45.0,
                description="Base diffusion model (14B parameters)",
                files_to_check=["diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]
            ),
            "wan_1.3b": ModelInfo(
                name="Wan2.1-T2V-1.3B",
                huggingface_repo="Wan-AI/Wan2.1-T2V-1.3B",
                local_path=str(base_path / "Wan2.1-T2V-1.3B"),
                size_gb=8.5,
                description="Base diffusion model (1.3B parameters)",
                files_to_check=["diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"]
            ),
            
            # OmniAvatar models
            "omniavatar_14b": ModelInfo(
                name="OmniAvatar-14B", 
                huggingface_repo="OmniAvatar/OmniAvatar-14B",
                local_path=str(base_path / "OmniAvatar-14B"),
                size_gb=2.5,
                description="OmniAvatar LoRA weights and audio conditioning (14B)",
                files_to_check=["config.json", "pytorch_model.pt"]
            ),
            "omniavatar_1.3b": ModelInfo(
                name="OmniAvatar-1.3B",
                huggingface_repo="OmniAvatar/OmniAvatar-1.3B", 
                local_path=str(base_path / "OmniAvatar-1.3B"),
                size_gb=1.2,
                description="OmniAvatar LoRA weights and audio conditioning (1.3B)",
                files_to_check=["config.json", "pytorch_model.pt"]
            ),
            
            # Audio model
            "wav2vec": ModelInfo(
                name="Wav2Vec2",
                huggingface_repo="facebook/wav2vec2-base-960h",
                local_path=str(base_path / "wav2vec2-base-960h"),
                size_gb=1.1,
                required=True,
                description="Audio encoder for speech processing",
                files_to_check=["pytorch_model.bin", "config.json", "vocab.json"]
            )
        }
    
    def get_models_for_size(self, model_size: ModelSize) -> List[str]:
        """Get model keys for a specific size variant"""
        if model_size == ModelSize.SMALL:
            return ["wan_1.3b", "omniavatar_1.3b", "wav2vec"]
        else:  # ModelSize.LARGE
            return ["wan_14b", "omniavatar_14b", "wav2vec"]
    
    def get_total_size(self, model_keys: List[str]) -> float:
        """Calculate total download size for given models"""
        return sum(self.models[key].size_gb for key in model_keys if key in self.models)

class DownloadProgress:
    """Track download progress with callbacks"""
    
    def __init__(self, model_name: str, total_size_gb: float):
        self.model_name = model_name
        self.total_size_gb = total_size_gb
        self.start_time = time.time()
        self.bytes_downloaded = 0
        self.is_complete = False
        self.error = None
        
    def update(self, bytes_downloaded: int):
        """Update progress"""
        self.bytes_downloaded = bytes_downloaded
        
    def get_progress_percent(self) -> float:
        """Get progress as percentage"""
        if self.total_size_gb == 0:
            return 0.0
        return min(100.0, (self.bytes_downloaded / (self.total_size_gb * 1024**3)) * 100)
    
    def get_speed_mbps(self) -> float:
        """Get download speed in MB/s"""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return (self.bytes_downloaded / (1024**2)) / elapsed
    
    def get_eta_minutes(self) -> float:
        """Get estimated time remaining in minutes"""
        speed_bps = self.bytes_downloaded / (time.time() - self.start_time) if time.time() > self.start_time else 0
        if speed_bps == 0:
            return float('inf')
        remaining_bytes = (self.total_size_gb * 1024**3) - self.bytes_downloaded
        return (remaining_bytes / speed_bps) / 60

class ModelDownloader:
    """Downloads OmniAvatar models with progress tracking"""
    
    def __init__(self, pretrained_models_path: Optional[Path] = None, python_exe: Optional[str] = None):
        if pretrained_models_path is None:
            pretrained_models_path = Path.cwd() / "pretrained_models"
        
        self.pretrained_models_path = pretrained_models_path
        self.registry = ModelRegistry(pretrained_models_path)
        self.python_exe = python_exe or sys.executable
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Progress tracking
        self.progress_callbacks: List[Callable] = []
        self.download_threads: Dict[str, threading.Thread] = {}
        
    def add_progress_callback(self, callback: Callable[[str, DownloadProgress], None]):
        """Add a progress callback function"""
        self.progress_callbacks.append(callback)
        
    def _notify_progress(self, model_name: str, progress: DownloadProgress):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(model_name, progress)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
    
    def check_model_exists(self, model_key: str) -> bool:
        """Check if a model is already downloaded and valid"""
        if model_key not in self.registry.models:
            return False
            
        model = self.registry.models[model_key]
        model_path = Path(model.local_path)
        
        if not model_path.exists():
            return False
            
        # Check for required files
        for file_name in model.files_to_check:
            if not (model_path / file_name).exists():
                self.logger.warning(f"Missing file in {model.name}: {file_name}")
                return False
                
        self.logger.info(f"✅ {model.name} already exists and is valid")
        return True
    
    def download_model(self, model_key: str, force_redownload: bool = False) -> bool:
        """Download a single model"""
        if model_key not in self.registry.models:
            self.logger.error(f"❌ Unknown model key: {model_key}")
            return False
            
        model = self.registry.models[model_key]
        
        # Check if already exists
        if not force_redownload and self.check_model_exists(model_key):
            return True
            
        self.logger.info(f"🔽 Downloading {model.name} ({model.size_gb:.1f}GB)")
        self.logger.info(f"📂 Destination: {model.local_path}")
        
        try:
            # Create progress tracker
            progress = DownloadProgress(model.name, model.size_gb)
            
            # Ensure pretrained_models directory exists
            self.pretrained_models_path.mkdir(exist_ok=True)
            
            # Build huggingface-cli download command
            cmd = [
                self.python_exe, "-m", "huggingface_hub", "download",
                model.huggingface_repo,
                "--local-dir", model.local_path,
                "--resume-download"  # Enable resume capability
            ]
            
            self.logger.info(f"Running: {' '.join(cmd)}")
            
            # Start download process
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Monitor progress
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                    
                if output:
                    line = output.strip()
                    self.logger.info(f"  {line}")
                    
                    # Update progress based on output parsing
                    if "downloading" in line.lower() or "%" in line:
                        # Simple progress notification
                        self._notify_progress(model.name, progress)
            
            return_code = process.poll()
            
            if return_code == 0:
                progress.is_complete = True
                self._notify_progress(model.name, progress)
                self.logger.info(f"✅ {model.name} download completed")
                
                # Verify download
                if self.check_model_exists(model_key):
                    return True
                else:
                    self.logger.error(f"❌ {model.name} download verification failed")
                    return False
            else:
                error_msg = f"Download failed with return code {return_code}"
                progress.error = error_msg
                self._notify_progress(model.name, progress)
                self.logger.error(f"❌ {error_msg}")
                return False
                
        except Exception as e:
            error_msg = f"Download error: {e}"
            self.logger.error(f"❌ {error_msg}")
            if 'progress' in locals():
                progress.error = error_msg
                self._notify_progress(model.name, progress)
            return False
    
    def download_models(self, model_keys: List[str], max_concurrent: int = 2) -> Dict[str, bool]:
        """Download multiple models with limited concurrency"""
        results = {}
        
        # Filter out models that already exist
        models_to_download = []
        for key in model_keys:
            if self.check_model_exists(key):
                results[key] = True
            else:
                models_to_download.append(key)
        
        if not models_to_download:
            self.logger.info("✅ All models already downloaded and valid")
            return results
        
        total_size = self.registry.get_total_size(models_to_download)
        self.logger.info(f"📊 Downloading {len(models_to_download)} models ({total_size:.1f}GB total)")
        
        # Download models with limited concurrency
        with ThreadPoolExecutor(max_workers=min(max_concurrent, len(models_to_download))) as executor:
            # Submit download tasks
            future_to_key = {
                executor.submit(self.download_model, key): key 
                for key in models_to_download
            }
            
            # Process completions
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    success = future.result()
                    results[key] = success
                    if success:
                        self.logger.info(f"✅ {self.registry.models[key].name} completed")
                    else:
                        self.logger.error(f"❌ {self.registry.models[key].name} failed")
                except Exception as e:
                    self.logger.error(f"❌ {key} download exception: {e}")
                    results[key] = False
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        self.logger.info(f"📊 Download summary: {successful}/{total} models successful")
        
        return results
    
    def download_for_model_size(self, model_size: ModelSize, max_concurrent: int = 2) -> bool:
        """Download all models for a specific size variant"""
        model_keys = self.registry.get_models_for_size(model_size)
        total_size = self.registry.get_total_size(model_keys)
        
        self.logger.info(f"🎯 Downloading {model_size.value} model variant")
        self.logger.info(f"📦 Models: {[self.registry.models[key].name for key in model_keys]}")
        self.logger.info(f"💾 Total size: {total_size:.1f}GB")
        
        results = self.download_models(model_keys, max_concurrent)
        
        # Check if all critical models downloaded successfully
        failed_models = [key for key, success in results.items() if not success]
        if failed_models:
            self.logger.error(f"❌ Failed to download: {[self.registry.models[key].name for key in failed_models]}")
            return False
        else:
            self.logger.info(f"✅ All {model_size.value} models downloaded successfully!")
            return True
    
    def get_download_status(self) -> Dict[str, Dict]:
        """Get status of all models"""
        status = {}
        
        for key, model in self.registry.models.items():
            exists = self.check_model_exists(key)
            status[key] = {
                'name': model.name,
                'exists': exists,
                'size_gb': model.size_gb,
                'path': model.local_path,
                'description': model.description
            }
            
        return status
    
    def cleanup_partial_downloads(self):
        """Clean up any partial or corrupted downloads"""
        self.logger.info("🧹 Cleaning up partial downloads...")
        
        for key, model in self.registry.models.items():
            model_path = Path(model.local_path)
            if model_path.exists() and not self.check_model_exists(key):
                self.logger.info(f"🗑️  Removing incomplete download: {model.name}")
                try:
                    import shutil
                    shutil.rmtree(model_path)
                except Exception as e:
                    self.logger.warning(f"Could not remove {model_path}: {e}")
    
    def estimate_download_time(self, model_keys: List[str], speed_mbps: float = 10.0) -> float:
        """Estimate download time in minutes"""
        total_size_gb = self.registry.get_total_size(model_keys)
        total_size_mb = total_size_gb * 1024
        return total_size_mb / speed_mbps / 60  # Convert to minutes

def main():
    """CLI entry point for model downloading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download OmniAvatar models")
    parser.add_argument("--size", choices=["1.3B", "14B"], default="1.3B", 
                       help="Model size variant to download")
    parser.add_argument("--models", nargs="+", 
                       help="Specific models to download (wan_14b, wan_1.3b, omniavatar_14b, omniavatar_1.3b, wav2vec)")
    parser.add_argument("--force", action="store_true", 
                       help="Force redownload existing models")
    parser.add_argument("--status", action="store_true", 
                       help="Show download status of all models")
    parser.add_argument("--cleanup", action="store_true", 
                       help="Clean up partial downloads")
    parser.add_argument("--path", type=Path, 
                       help="Custom path for pretrained_models directory")
    parser.add_argument("--max-concurrent", type=int, default=2,
                       help="Maximum concurrent downloads")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(pretrained_models_path=args.path)
    
    if args.status:
        status = downloader.get_download_status()
        print("\n📊 Model Download Status:")
        print("=" * 60)
        for key, info in status.items():
            status_icon = "✅" if info['exists'] else "❌"
            print(f"{status_icon} {info['name']}")
            print(f"   Size: {info['size_gb']:.1f}GB")
            print(f"   Path: {info['path']}")
            print(f"   Description: {info['description']}")
            print()
        return 0
    
    if args.cleanup:
        downloader.cleanup_partial_downloads()
        return 0
    
    if args.models:
        # Download specific models
        results = downloader.download_models(args.models, args.max_concurrent)
        success = all(results.values())
    else:
        # Download for model size
        model_size = ModelSize.SMALL if args.size == "1.3B" else ModelSize.LARGE
        success = downloader.download_for_model_size(model_size, args.max_concurrent)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())