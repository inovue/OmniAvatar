import os
from typing import Any, Dict, Tuple

class OmniAvatarConfig:
    """
    ComfyUI node for OmniAvatar configuration settings.
    This node organizes all OmniAvatar parameters into logical groups.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Model Paths
                "dit_path": ("STRING", {
                    "default": "pretrained_models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
                    "multiline": False,
                    "tooltip": "Path to DiT model files (comma-separated for multiple files)"
                }),
                "text_encoder_path": ("STRING", {
                    "default": "pretrained_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                    "multiline": False,
                    "tooltip": "Path to text encoder model"
                }),
                "vae_path": ("STRING", {
                    "default": "pretrained_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
                    "multiline": False,
                    "tooltip": "Path to VAE model"
                }),
                "wav2vec_path": ("STRING", {
                    "default": "pretrained_models/wav2vec2-base-960h",
                    "multiline": False,
                    "tooltip": "Path to Wav2Vec model for audio processing"
                }),
                "exp_path": ("STRING", {
                    "default": "pretrained_models/OmniAvatar-1.3B",
                    "multiline": False,
                    "tooltip": "Experiment path containing pytorch_model.pt"
                }),
                
                # Generation Settings
                "num_steps": ("INT", {
                    "default": 50,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Number of diffusion inference steps"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 4.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "CFG guidance scale"
                }),
                "negative_prompt": ("STRING", {
                    "default": "Vivid color tones, background/camera moving quickly, screen switching, subtitles and special effects, mutation, overexposed, static, blurred details, subtitles, style, work, painting, image, still, overall grayish, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fingers merging, motionless image, chaotic background, three legs, crowded background with many people, walking backward",
                    "multiline": True,
                    "tooltip": "Negative prompt for generation"
                }),
                "seq_len": ("INT", {
                    "default": 200,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Sequence length for generation"
                }),
                "overlap_frame": ("INT", {
                    "default": 13,
                    "min": 1,
                    "max": 100,
                    "step": 4,
                    "tooltip": "Frame overlap for long video generation (must be 1 + 4*n)"
                }),
                
                # Video/Audio Settings
                "fps": ("INT", {
                    "default": 25,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "tooltip": "Frames per second for video generation"
                }),
                "max_hw": ("COMBO", {
                    "default": 720,
                    "values": [720, 1280],
                    "tooltip": "Maximum height/width (720: 480p, 1280: 720p)"
                }),
                "max_tokens": ("INT", {
                    "default": 30000,
                    "min": 1000,
                    "max": 100000,
                    "step": 1000,
                    "tooltip": "Maximum tokens for generation"
                }),
                "sample_rate": ("INT", {
                    "default": 16000,
                    "min": 8000,
                    "max": 48000,
                    "step": 8000,
                    "tooltip": "Audio sample rate"
                }),
                "silence_duration_s": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.1,
                    "tooltip": "Silence duration in seconds for audio processing"
                }),
                
                # Technical Settings
                "dtype": ("COMBO", {
                    "default": "bf16",
                    "values": ["bf16", "fp16", "fp32"],
                    "tooltip": "Data type for model computations"
                }),
                "sp_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Sequence parallel size for distributed processing"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 2147483647,
                    "tooltip": "Random seed for reproducible generation"
                }),
            },
            
            "optional": {
                # Audio Processing
                "use_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable audio-driven video generation"
                }),
                "audio_scale": ("FLOAT", {
                    "default": None,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Audio CFG scale (uses guidance_scale if None)"
                }),
                
                # Image-to-Video Settings
                "i2v": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable image-to-video mode"
                }),
                "random_prefix_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use random prefix frames in i2v mode"
                }),
                
                # Training Architecture (LoRA)
                "train_architecture": ("COMBO", {
                    "default": None,
                    "values": [None, "lora"],
                    "tooltip": "Training architecture (set to 'lora' for LoRA models)"
                }),
                "lora_rank": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "LoRA rank (only used if train_architecture='lora')"
                }),
                "lora_alpha": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "LoRA alpha (only used if train_architecture='lora')"
                }),
                "lora_target_modules": ("STRING", {
                    "default": "q,k,v,o,ffn.0,ffn.2",
                    "multiline": False,
                    "tooltip": "Comma-separated target modules for LoRA"
                }),
                "init_lora_weights": ("COMBO", {
                    "default": "kaiming",
                    "values": ["kaiming", "gaussian", "zeros"],
                    "tooltip": "LoRA weight initialization method"
                }),
                
                # Advanced Performance Settings
                "use_fsdp": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Use Fully Sharded Data Parallel (for large models)"
                }),
                "num_persistent_param_in_dit": ("INT", {
                    "default": None,
                    "min": 0,
                    "max": 10000,
                    "step": 100,
                    "tooltip": "Number of persistent parameters in DiT (reduces VRAM usage)"
                }),
                "tea_cache_l1_thresh": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "TEA cache L1 threshold (higher = faster but lower quality)"
                }),
                
                # Debug and System
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable debug mode"
                }),
            }
        }
    
    RETURN_TYPES = ("OMNIAVATAR_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "create_config"
    CATEGORY = "OmniAvatar"
    DESCRIPTION = "Configuration node for OmniAvatar video generation parameters"
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate configuration parameters"""
        errors = []
        
        # Check model paths exist
        required_paths = ["dit_path", "text_encoder_path", "vae_path", "exp_path"]
        for path_key in required_paths:
            paths = config.get(path_key, "")
            if not paths:
                errors.append(f"Required path {path_key} is empty")
                continue
                
            # Handle comma-separated paths (for dit_path)
            path_list = [p.strip() for p in paths.split(",")]
            for path in path_list:
                if not os.path.exists(path):
                    errors.append(f"Path {path_key}: {path} does not exist")
        
        # Check wav2vec path if audio is enabled
        if config.get("use_audio", True):
            wav2vec_path = config.get("wav2vec_path", "")
            if not wav2vec_path or not os.path.exists(wav2vec_path):
                errors.append(f"Wav2Vec path required for audio processing: {wav2vec_path}")
        
        # Validate overlap_frame format (must be 1 + 4*n)
        overlap_frame = config.get("overlap_frame", 13)
        if (overlap_frame - 1) % 4 != 0:
            errors.append(f"overlap_frame must be 1 + 4*n, got {overlap_frame}")
        
        # Validate image sizes based on max_hw
        max_hw = config.get("max_hw", 720)
        if max_hw not in [720, 1280]:
            errors.append(f"max_hw must be 720 or 1280, got {max_hw}")
        
        # Add image sizes to config based on max_hw
        if max_hw == 720:
            config["image_sizes_720"] = [[400, 720], [720, 720], [720, 400]]
        else:
            config["image_sizes_1280"] = [
                [720, 720], [528, 960], [960, 528], [720, 1280], [1280, 720]
            ]
        
        return len(errors) == 0, "; ".join(errors) if errors else "Configuration valid"
    
    def create_config(self, **kwargs):
        """Create configuration object from input parameters"""
        
        # Create configuration dictionary
        config = {}
        
        # Add all parameters to config
        for key, value in kwargs.items():
            config[key] = value
        
        # Add computed values
        config["rank"] = 0  # Default for single GPU
        config["local_rank"] = 0
        config["world_size"] = 1
        config["num_nodes"] = 1
        config["device"] = "cuda:0"
        
        # Set default audio_scale to guidance_scale if not provided
        if config.get("audio_scale") is None:
            config["audio_scale"] = config.get("guidance_scale", 4.5)
        
        # Validate configuration
        is_valid, message = self.validate_config(config)
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {message}")
        
        print(f"[OmniAvatar Config] Configuration created successfully")
        print(f"[OmniAvatar Config] Models: DiT={os.path.exists(config['dit_path'].split(',')[0])}, "
              f"VAE={os.path.exists(config['vae_path'])}, "
              f"TextEnc={os.path.exists(config['text_encoder_path'])}")
        
        return (config,)