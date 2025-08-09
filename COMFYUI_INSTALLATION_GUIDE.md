# OmniAvatar ComfyUI Automated Installation Guide

This guide covers the automated installation system for OmniAvatar ComfyUI custom nodes.

## 🚀 Quick Start

### One-Command Installation

```bash
# Run from the OmniAvatar root directory
python setup_comfyui.py
```

This will:
- ✅ Validate your system requirements
- 📦 Install PyTorch and dependencies
- ⬇️ Download the 1.3B model (recommended for most users)
- 🔗 Integrate with ComfyUI
- 🔍 Verify the installation

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.8+
- **GPU**: CUDA-compatible with 8GB+ VRAM
- **Disk Space**: 100GB+ free space
- **OS**: Windows, Linux, or macOS with CUDA support

### Recommended Requirements
- **GPU**: 21GB+ VRAM for 14B model
- **RAM**: 16GB+ system memory
- **Internet**: Stable connection for model downloads

## 🛠️ Installation Modes

### Quick Installation (Default)
```bash
python setup_comfyui.py --mode quick
```
- 1.3B model (faster, lower VRAM)
- Essential dependencies only
- ~10GB download

### Full Installation
```bash
python setup_comfyui.py --mode full
```
- 14B model (higher quality, more VRAM)
- All optional dependencies
- ~50GB download

### Custom Installation
```bash
python setup_comfyui.py --mode custom
```
- Interactive selection of components
- Choose model size and features

## 🔧 Advanced Options

### Command Line Options

```bash
# Specific model size
python setup_comfyui.py --model-size 14B

# Skip optional dependencies
python setup_comfyui.py --no-optional

# Force reinstall everything
python setup_comfyui.py --force

# Custom models directory
python setup_comfyui.py --models-path /path/to/models

# Development mode (symlinks instead of copying)
python setup_comfyui.py --symlink

# Skip specific phases
python setup_comfyui.py --skip-validation  # Skip system checks
python setup_comfyui.py --skip-models      # Skip model download
python setup_comfyui.py --skip-dependencies # Skip pip installs
```

### Individual Component Installation

```bash
# System validation only
python -m comfyui_nodes.installer.system_validator

# Install dependencies only
python -m comfyui_nodes.installer.dependency_manager

# Download models only
python -m comfyui_nodes.installer.model_downloader --size 1.3B

# ComfyUI integration only
python -m comfyui_nodes.installer.comfyui_integrator --source comfyui_nodes/
```

## 🎯 Model Information

### 1.3B Model (Recommended)
- **Size**: ~10GB total download
- **VRAM**: 8GB minimum, 12GB recommended
- **Speed**: Faster inference
- **Quality**: Good for most use cases

### 14B Model (High Quality)
- **Size**: ~50GB total download  
- **VRAM**: 21GB minimum, 36GB recommended
- **Speed**: Slower inference
- **Quality**: Highest quality output

### Model Components
- **Base Model**: Wan2.1-T2V diffusion transformer
- **OmniAvatar**: LoRA weights and audio conditioning
- **Audio Model**: Wav2Vec2 for speech processing
- **VAE**: Video autoencoder for latent space
- **Text Encoder**: T5 for text understanding

## 📂 Directory Structure

After installation:

```
OmniAvatar/
├── setup_comfyui.py                   # Main setup script
├── comfyui_nodes/                     # ComfyUI node source
│   ├── __init__.py                    # Node registration
│   ├── omniavatarconfig.py            # Config node
│   ├── omniavatarInference.py         # Inference node
│   ├── requirements.txt               # Python dependencies
│   └── installer/                     # Installation system
│       ├── main_installer.py          # Main orchestrator
│       ├── system_validator.py        # System validation
│       ├── dependency_manager.py      # Dependency installation
│       ├── model_downloader.py        # Model downloading
│       └── comfyui_integrator.py      # ComfyUI integration
└── pretrained_models/                 # Downloaded models
    ├── Wan2.1-T2V-1.3B/              # Base model
    ├── OmniAvatar-1.3B/               # OmniAvatar weights
    └── wav2vec2-base-960h/            # Audio model
```

ComfyUI integration creates:
```
ComfyUI/custom_nodes/omniavatarnodes/   # Installed nodes
├── __init__.py                        # Node registration
├── omniavatarconfig.py                # Config node
├── omniavatarInference.py             # Inference node
├── requirements.txt                   # Dependencies
├── install.py                         # ComfyUI Manager script
├── uninstall.py                       # Cleanup script
└── config/                            # Configuration files
    └── default_config.json            # Default settings
```

## 🔍 Troubleshooting

### Installation Issues

**CUDA Not Available**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
python setup_comfyui.py --force --skip-models
```

**Insufficient VRAM**
```bash
# Use 1.3B model instead
python setup_comfyui.py --model-size 1.3B

# Or use memory optimization
# (Configure in OmniAvatarConfig node)
```

**Download Failures**
```bash
# Retry with better connection
python setup_comfyui.py --force

# Or download specific models
python -m comfyui_nodes.installer.model_downloader --size 1.3B
```

**ComfyUI Not Detected**
```bash
# Set ComfyUI path manually
export COMFYUI_PATH=/path/to/ComfyUI
python setup_comfyui.py

# Or validate ComfyUI installation
python -m comfyui_nodes.installer.comfyui_integrator --validate-only
```

### Runtime Issues

**Nodes Not Appearing in ComfyUI**
1. Restart ComfyUI completely
2. Check ComfyUI console for error messages
3. Verify installation:
   ```bash
   python -c "from comfyui_nodes import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"
   ```

**Out of Memory Errors**
1. Reduce `seq_len` in OmniAvatarConfig
2. Enable `use_fsdp` in OmniAvatarConfig  
3. Set `num_persistent_param_in_dit` to limit memory usage
4. Switch to 1.3B model

**Model Loading Errors**
1. Verify model paths in OmniAvatarConfig
2. Check that all models downloaded successfully:
   ```bash
   python -m comfyui_nodes.installer.model_downloader --status
   ```

## 📊 Performance Optimization

### Memory Optimization
- **1.3B Model**: Use for development and testing
- **FSDP**: Enable `use_fsdp` for distributed memory usage
- **Persistent Params**: Set `num_persistent_param_in_dit` to limit VRAM
- **TEA Cache**: Enable `tea_cache_l1_thresh` for speed vs quality trade-off

### Speed Optimization  
- **Reduce Steps**: Lower `num_steps` (20-50 range)
- **TEA Cache**: Set `tea_cache_l1_thresh` to 0.05-0.15
- **Multi-GPU**: Use multiple GPUs if available

### Quality Enhancement
- **More Steps**: Increase `num_steps` to 100+
- **Higher Resolution**: Increase video resolution settings
- **Better Prompts**: Use detailed, specific descriptions

## 🆘 Support

### Log Files
Installation logs are displayed during setup. For detailed debugging:
```bash
python setup_comfyui.py 2>&1 | tee installation.log
```

### System Information
```bash
# Get system details
python -m comfyui_nodes.installer.system_validator

# Check dependencies
python -m comfyui_nodes.installer.dependency_manager --verify-only

# Model status
python -m comfyui_nodes.installer.model_downloader --status
```

### Common Solutions
1. **Update GPU drivers** if CUDA issues persist
2. **Increase virtual memory** if system runs out of RAM
3. **Use SSD storage** for better model loading performance
4. **Check firewall settings** if downloads fail
5. **Run as administrator** if permission issues occur

### Getting Help
- Check the [OmniAvatar GitHub Issues](https://github.com/Omni-Avatar/OmniAvatar/issues)
- Review ComfyUI console logs for specific errors
- Include system information when reporting issues

## ✅ Verification Checklist

After installation, verify:
- [ ] ComfyUI restarts without errors
- [ ] OmniAvatar nodes appear in node menu
- [ ] OmniAvatarConfig node loads correctly
- [ ] Model paths are correctly configured
- [ ] GPU/CUDA detection works
- [ ] Test generation works (simple prompt)

## 🔄 Updates and Maintenance

### Updating Models
```bash
# Download latest models
python -m comfyui_nodes.installer.model_downloader --force --size 1.3B
```

### Updating Dependencies  
```bash
# Update Python packages
python -m comfyui_nodes.installer.dependency_manager --force
```

### Clean Installation
```bash
# Remove existing installation
rm -rf ComfyUI/custom_nodes/omniavatarnodes/
rm -rf pretrained_models/

# Reinstall
python setup_comfyui.py --force
```

---

## 💡 Tips for Success

1. **Start with 1.3B model** for testing and learning
2. **Ensure stable internet** for model downloads  
3. **Have patience** - initial setup can take 30-60 minutes
4. **Monitor VRAM usage** and adjust settings accordingly
5. **Use detailed prompts** for better generation results
6. **Restart ComfyUI** after any configuration changes