# OmniAvatar ComfyUI Nodes

ComfyUI integration for OmniAvatar video generation model, supporting text-to-video, image-to-video, and audio-driven video generation.

## Features

- **Text-to-Video**: Generate videos from text prompts
- **Image-to-Video**: Animate static images with motion
- **Audio-Driven**: Synchronize video generation with audio input
- **Flexible Configuration**: Comprehensive parameter control through config node
- **GPU Optimization**: Efficient VRAM management and performance optimization
- **Error Handling**: Robust error reporting and dependency validation

## Nodes

### OmniAvatarConfig
Configuration node that centralizes all OmniAvatar parameters:

- **Model Paths**: DiT, VAE, Text Encoder, Wav2Vec model locations
- **Generation Settings**: Steps, guidance scale, negative prompts
- **Video/Audio Settings**: FPS, resolution, audio processing options
- **Technical Settings**: Data types, LoRA configuration, performance options
- **Advanced Settings**: FSDP, TEA cache, memory optimization

### OmniAvatarInference
Main inference node for video generation:

**Required Inputs:**
- `config`: Configuration from OmniAvatarConfig node
- `prompt`: Text description for video generation

**Optional Inputs:**
- `image`: Input image for image-to-video generation
- `audio`: Audio input for audio-driven generation
- `seq_len`, `height`, `width`: Override dimension settings
- `num_steps`, `guidance_scale`: Override generation parameters

**Output:**
- `video`: Generated video tensor in ComfyUI format

## Installation

### Prerequisites

- **Python 3.8+**
- **PyTorch 2.0+** with CUDA support
- **CUDA GPU** with 8GB+ VRAM recommended
- **ComfyUI** installed and working

### Step 1: Install OmniAvatar

```bash
# Clone OmniAvatar repository
git clone https://github.com/yourorg/OmniAvatar
cd OmniAvatar

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Step 2: Install Additional Dependencies

```bash
# Core dependencies
pip install transformers peft librosa soundfile

# Optional for distributed processing
pip install xfuser
```

### Step 3: Install ComfyUI Nodes

```bash
# Copy nodes to ComfyUI custom_nodes directory
cp -r comfyui_nodes /path/to/ComfyUI/custom_nodes/omniavatarnodes

# Or create symlink for development
ln -s /path/to/OmniAvatar/comfyui_nodes /path/to/ComfyUI/custom_nodes/omniavatarnodes
```

### Step 4: Download Model Weights

Download the required model weights and update paths in the configuration:

- **DiT Model**: Diffusion transformer weights
- **Text Encoder**: T5 text encoder weights  
- **VAE**: Video autoencoder weights
- **Audio Model**: Wav2Vec2 weights for audio processing
- **Experiment**: Trained OmniAvatar checkpoint (pytorch_model.pt)

## Usage

### Basic Text-to-Video

1. Add **OmniAvatarConfig** node
2. Configure model paths and basic settings
3. Add **OmniAvatarInference** node
4. Connect config output to inference input
5. Set text prompt: "A beautiful woman talking"
6. Run generation

### Image-to-Video

1. Enable `i2v` mode in config
2. Load image using ComfyUI image loader
3. Connect image to inference node
4. Generate animated video from static image

### Audio-Driven Generation  

1. Enable `use_audio` in config
2. Load audio file using ComfyUI audio loader
3. Connect audio to inference node
4. Generate video synchronized with audio

### Advanced Configuration

**Performance Optimization:**
- Reduce `max_tokens` for lower VRAM usage
- Use `num_persistent_param_in_dit` to limit memory
- Enable `tea_cache_l1_thresh` for faster inference

**Quality Control:**
- Increase `num_steps` for better quality
- Adjust `guidance_scale` for prompt adherence
- Use detailed negative prompts

**LoRA Fine-tuning:**
- Set `train_architecture` to "lora"
- Configure `lora_rank` and `lora_alpha`
- Point `exp_path` to LoRA checkpoint

## Configuration Reference

### Model Paths
```
dit_path: "pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"
text_encoder_path: "pretrained_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth"
vae_path: "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"
wav2vec_path: "pretrained_models/wav2vec2-base-960h"
exp_path: "pretrained_models/OmniAvatar-1.3B"
```

### Generation Settings
```
num_steps: 50               # Inference steps (quality vs speed)
guidance_scale: 4.5         # CFG scale (prompt adherence)
audio_scale: 4.5           # Audio CFG scale
seq_len: 200               # Video length in frames
fps: 25                    # Frames per second
```

### Technical Settings
```
dtype: "bf16"              # Precision (bf16/fp16/fp32)
max_hw: 720               # Resolution (720p/1080p)
max_tokens: 30000         # Context length
seed: 42                  # Random seed
```

## Troubleshooting

### Dependency Issues
```bash
# Check dependencies
python -c "from comfyui_nodes.error_handling import check_dependencies; print(check_dependencies())"

# Install missing packages
pip install transformers peft librosa soundfile
```

### CUDA Out of Memory
- Reduce `max_tokens` (e.g., 15000 → 10000)
- Lower `seq_len` for shorter videos
- Use `fp16` instead of `bf16`
- Enable `num_persistent_param_in_dit`

### Model Loading Errors
- Verify all model paths exist and are correct
- Check `exp_path` contains `pytorch_model.pt`
- Ensure sufficient disk space for model loading
- Validate file permissions

### Generation Failures
- Check prompt length and complexity
- Verify input image/audio formats
- Monitor GPU memory usage
- Review error logs for specific issues

## Performance Tips

### Memory Optimization
- Use `num_persistent_param_in_dit` to reduce VRAM
- Enable gradient checkpointing if available
- Close other GPU applications

### Speed Optimization  
- Use `tea_cache_l1_thresh` for faster inference (quality trade-off)
- Reduce `num_steps` if quality acceptable
- Use `fp16` precision on compatible hardware

### Quality Enhancement
- Increase `num_steps` to 100+ for best quality
- Use detailed prompts with specific descriptions
- Experiment with `guidance_scale` values
- Craft effective negative prompts

## Limitations

- Requires CUDA GPU with substantial VRAM
- Video length limited by GPU memory
- Audio synchronization depends on model training
- Generation time scales with video length and quality settings

## Support

For issues and questions:
1. Check this README and troubleshooting section
2. Verify dependency installation and versions
3. Review ComfyUI console logs for error details
4. Report bugs with system information and error logs