"""
System Validator for OmniAvatar ComfyUI Installation
Validates system requirements before installation.
"""
import os
import sys
import shutil
import subprocess
import psutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@dataclass
class SystemRequirements:
    """System requirement specifications"""
    min_python_version: Tuple[int, int] = (3, 8)
    min_vram_gb: int = 8
    recommended_vram_gb: int = 21
    min_disk_space_gb: int = 100
    required_cuda_version: str = "11.8+"
    required_pytorch_version: str = "2.0+"

@dataclass
class ValidationResult:
    """System validation result"""
    passed: bool
    warnings: List[str]
    errors: List[str]
    system_info: Dict[str, str]
    recommendations: List[str]

class SystemValidator:
    """Validates system requirements for OmniAvatar installation"""
    
    def __init__(self):
        self.requirements = SystemRequirements()
        
    def validate_system(self) -> ValidationResult:
        """Perform comprehensive system validation"""
        warnings = []
        errors = []
        recommendations = []
        system_info = {}
        
        # Python version check
        python_result = self._check_python_version()
        system_info.update(python_result['info'])
        if not python_result['passed']:
            errors.extend(python_result['errors'])
        
        # CUDA availability check
        cuda_result = self._check_cuda_availability()
        system_info.update(cuda_result['info'])
        if not cuda_result['passed']:
            errors.extend(cuda_result['errors'])
        else:
            warnings.extend(cuda_result['warnings'])
            
        # GPU VRAM check
        vram_result = self._check_gpu_memory()
        system_info.update(vram_result['info'])
        if vram_result['warnings']:
            warnings.extend(vram_result['warnings'])
        recommendations.extend(vram_result['recommendations'])
        
        # Disk space check
        disk_result = self._check_disk_space()
        system_info.update(disk_result['info'])
        if not disk_result['passed']:
            errors.extend(disk_result['errors'])
        elif disk_result['warnings']:
            warnings.extend(disk_result['warnings'])
            
        # PyTorch compatibility check
        if TORCH_AVAILABLE:
            pytorch_result = self._check_pytorch_compatibility()
            system_info.update(pytorch_result['info'])
            warnings.extend(pytorch_result['warnings'])
            recommendations.extend(pytorch_result['recommendations'])
        else:
            warnings.append("PyTorch not installed - will be installed during setup")
            recommendations.append("PyTorch 2.4.0 with CUDA 12.4 will be installed")
        
        # ComfyUI detection
        comfyui_result = self._detect_comfyui()
        system_info.update(comfyui_result['info'])
        if comfyui_result['warnings']:
            warnings.extend(comfyui_result['warnings'])
        recommendations.extend(comfyui_result['recommendations'])
        
        passed = len(errors) == 0
        
        return ValidationResult(
            passed=passed,
            warnings=warnings,
            errors=errors,
            system_info=system_info,
            recommendations=recommendations
        )
    
    def _check_python_version(self) -> Dict:
        """Check Python version compatibility"""
        current_version = sys.version_info[:2]
        min_version = self.requirements.min_python_version
        
        info = {
            'python_version': f"{current_version[0]}.{current_version[1]}.{sys.version_info[2]}",
            'python_executable': sys.executable
        }
        
        if current_version >= min_version:
            return {
                'passed': True,
                'errors': [],
                'info': info
            }
        else:
            return {
                'passed': False,
                'errors': [f"Python {min_version[0]}.{min_version[1]}+ required, found {current_version[0]}.{current_version[1]}"],
                'info': info
            }
    
    def _check_cuda_availability(self) -> Dict:
        """Check CUDA availability and version"""
        warnings = []
        errors = []
        info = {}
        
        # Check nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                info['nvidia_smi'] = 'Available'
                # Extract CUDA version from nvidia-smi output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        info['cuda_driver_version'] = cuda_version
                        break
            else:
                errors.append("nvidia-smi not available - NVIDIA GPU drivers may not be installed")
                info['nvidia_smi'] = 'Not available'
        except (subprocess.TimeoutExpired, FileNotFoundError):
            errors.append("nvidia-smi command failed - NVIDIA GPU drivers required")
            info['nvidia_smi'] = 'Not available'
        
        # Check CUDA runtime via PyTorch if available
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                info['torch_cuda'] = 'Available'
                info['torch_cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                info['current_gpu'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'None'
            else:
                warnings.append("PyTorch CUDA support not available")
                info['torch_cuda'] = 'Not available'
        else:
            info['torch_cuda'] = 'PyTorch not installed'
        
        passed = len(errors) == 0
        
        return {
            'passed': passed,
            'errors': errors,
            'warnings': warnings,
            'info': info
        }
    
    def _check_gpu_memory(self) -> Dict:
        """Check GPU VRAM availability"""
        warnings = []
        recommendations = []
        info = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                # Get GPU memory info
                total_memory = torch.cuda.get_device_properties(0).total_memory
                total_gb = total_memory / (1024**3)
                info['total_vram_gb'] = round(total_gb, 1)
                
                # Check against requirements
                min_vram = self.requirements.min_vram_gb
                recommended_vram = self.requirements.recommended_vram_gb
                
                if total_gb < min_vram:
                    warnings.append(f"Low VRAM: {total_gb:.1f}GB available, {min_vram}GB minimum required")
                    recommendations.append("Consider using 1.3B model instead of 14B model")
                    recommendations.append("Use memory optimization settings (num_persistent_param_in_dit)")
                elif total_gb < recommended_vram:
                    warnings.append(f"VRAM: {total_gb:.1f}GB available, {recommended_vram}GB recommended for optimal performance")
                    recommendations.append("Consider memory optimization settings for better performance")
                else:
                    info['vram_status'] = 'Optimal'
                    recommendations.append("VRAM sufficient for 14B model with full settings")
                    
            except Exception as e:
                warnings.append(f"Could not determine GPU memory: {e}")
                info['vram_status'] = 'Unknown'
        else:
            warnings.append("GPU memory information not available")
            info['vram_status'] = 'Unknown'
            recommendations.append("Ensure CUDA-compatible GPU is available")
        
        return {
            'warnings': warnings,
            'recommendations': recommendations,
            'info': info
        }
    
    def _check_disk_space(self) -> Dict:
        """Check available disk space"""
        current_dir = Path.cwd()
        disk_usage = psutil.disk_usage(current_dir)
        available_gb = disk_usage.free / (1024**3)
        
        info = {
            'available_disk_gb': round(available_gb, 1),
            'installation_path': str(current_dir)
        }
        
        min_space = self.requirements.min_disk_space_gb
        
        if available_gb < min_space:
            return {
                'passed': False,
                'errors': [f"Insufficient disk space: {available_gb:.1f}GB available, {min_space}GB required"],
                'warnings': [],
                'info': info
            }
        elif available_gb < min_space * 1.2:  # Warning if less than 120GB
            return {
                'passed': True,
                'errors': [],
                'warnings': [f"Low disk space: {available_gb:.1f}GB available"],
                'info': info
            }
        else:
            info['disk_status'] = 'Sufficient'
            return {
                'passed': True,
                'errors': [],
                'warnings': [],
                'info': info
            }
    
    def _check_pytorch_compatibility(self) -> Dict:
        """Check PyTorch version and CUDA compatibility"""
        warnings = []
        recommendations = []
        info = {}
        
        if TORCH_AVAILABLE:
            info['pytorch_version'] = torch.__version__
            info['pytorch_cuda_available'] = torch.cuda.is_available()
            info['pytorch_cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'Not available'
            
            # Check version compatibility
            pytorch_version = torch.__version__.split('+')[0]  # Remove +cu118 suffix if present
            version_parts = [int(x) for x in pytorch_version.split('.')]
            
            if version_parts[0] < 2:
                warnings.append(f"PyTorch {pytorch_version} detected, 2.0+ recommended")
                recommendations.append("Consider upgrading PyTorch for better performance")
            
            # Check CUDA version compatibility
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version and cuda_version < "11.8":
                    warnings.append(f"CUDA {cuda_version} detected, 11.8+ recommended")
        
        return {
            'warnings': warnings,
            'recommendations': recommendations,
            'info': info
        }
    
    def _detect_comfyui(self) -> Dict:
        """Detect ComfyUI installation"""
        warnings = []
        recommendations = []
        info = {}
        
        # Search for ComfyUI in common locations
        possible_paths = [
            Path.cwd() / '..' / '..' / 'ComfyUI',  # If running from custom_nodes
            Path.cwd() / 'ComfyUI',  # If ComfyUI is in current directory
            Path.cwd(),  # If we are in ComfyUI directory
        ]
        
        # Add environment variable path if set
        if 'COMFYUI_PATH' in os.environ:
            possible_paths.insert(0, Path(os.environ['COMFYUI_PATH']))
        
        comfyui_path = None
        for path in possible_paths:
            if self._is_comfyui_directory(path):
                comfyui_path = path.resolve()
                break
        
        if comfyui_path:
            info['comfyui_path'] = str(comfyui_path)
            info['comfyui_detected'] = 'Yes'
            
            # Check for custom_nodes directory
            custom_nodes_path = comfyui_path / 'custom_nodes'
            if custom_nodes_path.exists():
                info['custom_nodes_path'] = str(custom_nodes_path)
                recommendations.append(f"Will install to: {custom_nodes_path}")
            else:
                warnings.append("custom_nodes directory not found in ComfyUI installation")
                
            # Check for ComfyUI Manager
            manager_path = custom_nodes_path / 'ComfyUI-Manager'
            if manager_path.exists():
                info['comfyui_manager'] = 'Installed'
                recommendations.append("ComfyUI Manager detected - can be used for node management")
            else:
                info['comfyui_manager'] = 'Not installed'
                recommendations.append("Consider installing ComfyUI Manager for easier node management")
        else:
            warnings.append("ComfyUI installation not detected in common locations")
            recommendations.append("Set COMFYUI_PATH environment variable if ComfyUI is in a custom location")
            info['comfyui_detected'] = 'No'
        
        return {
            'warnings': warnings,
            'recommendations': recommendations,
            'info': info
        }
    
    def _is_comfyui_directory(self, path: Path) -> bool:
        """Check if a directory looks like a ComfyUI installation"""
        if not path.exists():
            return False
            
        # Look for key ComfyUI files
        comfyui_indicators = [
            'main.py',
            'comfy',
            'nodes.py',
            'model_management.py'
        ]
        
        for indicator in comfyui_indicators:
            if (path / indicator).exists():
                return True
        
        return False
    
    def print_validation_report(self, result: ValidationResult) -> None:
        """Print formatted validation report"""
        print("=" * 60)
        print("🔍 OmniAvatar Installation System Validation")
        print("=" * 60)
        
        # Overall status
        if result.passed:
            print("✅ System validation PASSED")
        else:
            print("❌ System validation FAILED")
        
        print()
        
        # System information
        if result.system_info:
            print("📋 System Information:")
            for key, value in result.system_info.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
            print()
        
        # Errors
        if result.errors:
            print("❌ Critical Issues (must be resolved):")
            for error in result.errors:
                print(f"  • {error}")
            print()
        
        # Warnings  
        if result.warnings:
            print("⚠️  Warnings:")
            for warning in result.warnings:
                print(f"  • {warning}")
            print()
        
        # Recommendations
        if result.recommendations:
            print("💡 Recommendations:")
            for rec in result.recommendations:
                print(f"  • {rec}")
            print()
        
        if not result.passed:
            print("🛠️  Please resolve the critical issues before proceeding with installation.")
        else:
            print("🚀 System ready for OmniAvatar installation!")
        
        print("=" * 60)

def main():
    """CLI entry point for system validation"""
    validator = SystemValidator()
    result = validator.validate_system()
    validator.print_validation_report(result)
    
    return 0 if result.passed else 1

if __name__ == "__main__":
    exit(main())