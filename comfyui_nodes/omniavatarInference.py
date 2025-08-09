import os
import tempfile
import torch
import numpy as np
from typing import Optional, Union, Dict, Any, Tuple
import warnings

from .inference_pipeline import SimplifiedWanInferencePipeline


class OmniAvatarInference:
    """
    ComfyUI node for OmniAvatar video generation inference.
    Accepts configuration from OmniAvatarConfig and generates videos from prompts.
    """
    
    def __init__(self):
        self.pipeline = None
        self.current_config_hash = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": ("OMNIAVATAR_CONFIG", {
                    "tooltip": "Configuration from OmniAvatarConfig node"
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful woman talking",
                    "tooltip": "Text prompt for video generation"
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Input image for image-to-video generation (requires i2v=True in config)"
                }),
                "audio": ("AUDIO", {
                    "tooltip": "Input audio for audio-driven video generation (requires use_audio=True in config)"
                }),
                "seq_len": ("INT", {
                    "default": None,
                    "min": 1,
                    "max": 1000,
                    "step": 1,
                    "tooltip": "Override sequence length from config"
                }),
                "height": ("INT", {
                    "default": 720,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Video height (used when no input image)"
                }),
                "width": ("INT", {
                    "default": 720,
                    "min": 256,
                    "max": 2048,
                    "step": 64,
                    "tooltip": "Video width (used when no input image)"
                }),
                # Generation parameter overrides
                "num_steps": ("INT", {
                    "default": None,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                    "tooltip": "Override number of inference steps"
                }),
                "guidance_scale": ("FLOAT", {
                    "default": None,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Override CFG guidance scale"
                }),
                "audio_scale": ("FLOAT", {
                    "default": None,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1,
                    "tooltip": "Override audio CFG scale"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Override negative prompt from config"
                }),
                "overlap_frame": ("INT", {
                    "default": None,
                    "min": 1,
                    "max": 100,
                    "step": 4,
                    "tooltip": "Override overlap frame (must be 1 + 4*n)"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)  # ComfyUI expects IMAGE format for video
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_video"
    CATEGORY = "OmniAvatar"
    DESCRIPTION = "Generate videos using OmniAvatar with text, optional image, and optional audio input"
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Create a hash of config to detect changes"""
        import hashlib
        import json
        # Create a sorted JSON string of relevant model paths and settings
        relevant_keys = [
            'dit_path', 'text_encoder_path', 'vae_path', 'wav2vec_path', 'exp_path',
            'dtype', 'train_architecture', 'lora_rank', 'lora_alpha', 'use_audio', 'i2v'
        ]
        relevant_config = {k: config.get(k) for k in relevant_keys if k in config}
        config_str = json.dumps(relevant_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _ensure_pipeline_loaded(self, config: Dict[str, Any]):
        """Ensure pipeline is loaded with current config"""
        config_hash = self._hash_config(config)
        
        if self.pipeline is None or self.current_config_hash != config_hash:
            print("[OmniAvatar] Loading pipeline with new configuration...")
            try:
                self.pipeline = SimplifiedWanInferencePipeline(config)
                self.current_config_hash = config_hash
                print("[OmniAvatar] Pipeline loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load OmniAvatar pipeline: {e}")
    
    def _convert_image_to_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Convert ComfyUI image format to pipeline format"""
        if image is None:
            return None
        
        # ComfyUI images are typically (B, H, W, C) in [0, 1] range
        if image.dim() == 4 and image.shape[0] == 1:
            image = image[0]  # Remove batch dimension
        
        # Convert from (H, W, C) to (C, H, W)
        if image.shape[-1] in [1, 3, 4]:  # Last dimension is channels
            image = image.permute(2, 0, 1)
        
        # Ensure RGB (3 channels)
        if image.shape[0] == 4:  # RGBA
            image = image[:3]  # Drop alpha channel
        elif image.shape[0] == 1:  # Grayscale
            image = image.repeat(3, 1, 1)  # Convert to RGB
        
        # Ensure values are in [0, 1] range
        image = torch.clamp(image, 0, 1)
        
        return image
    
    def _save_audio_to_temp_file(self, audio: Dict[str, Any]) -> Optional[str]:
        """Save ComfyUI audio to temporary file"""
        if audio is None:
            return None
        
        try:
            # ComfyUI audio format: {'waveform': tensor, 'sample_rate': int}
            waveform = audio.get('waveform')
            sample_rate = audio.get('sample_rate', 16000)
            
            if waveform is None:
                return None
            
            # Convert to numpy and ensure correct format
            if isinstance(waveform, torch.Tensor):
                waveform = waveform.detach().cpu().numpy()
            
            # Ensure mono audio
            if waveform.ndim == 2:
                waveform = waveform.mean(axis=0)  # Convert stereo to mono
            
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Save using soundfile or wave
            try:
                import soundfile as sf
                sf.write(temp_path, waveform, sample_rate)
            except ImportError:
                import wave
                with wave.open(temp_path, 'w') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    # Convert float to int16
                    audio_int16 = (waveform * 32767).astype(np.int16)
                    wav_file.writeframes(audio_int16.tobytes())
            
            return temp_path
            
        except Exception as e:
            warnings.warn(f"Failed to save audio to temporary file: {e}")
            return None
    
    def _convert_video_to_comfyui_format(self, video: torch.Tensor) -> torch.Tensor:
        """Convert pipeline output to ComfyUI image format"""
        # Pipeline returns video as (B, T, C, H, W) or similar
        # ComfyUI expects (T, H, W, C) for image sequences
        
        if video.dim() == 5:  # (B, T, C, H, W)
            video = video[0]  # Remove batch dimension -> (T, C, H, W)
        
        # Convert from (T, C, H, W) to (T, H, W, C)
        video = video.permute(0, 2, 3, 1)
        
        # Ensure values are in [0, 1] range
        video = torch.clamp(video, 0, 1)
        
        return video
    
    def _cleanup_temp_files(self, *temp_paths):
        """Clean up temporary files"""
        for temp_path in temp_paths:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass  # Ignore cleanup errors
    
    def generate_video(
        self,
        config: Dict[str, Any],
        prompt: str,
        image: Optional[torch.Tensor] = None,
        audio: Optional[Dict[str, Any]] = None,
        seq_len: Optional[int] = None,
        height: int = 720,
        width: int = 720,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        audio_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        overlap_frame: Optional[int] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Generate video using OmniAvatar pipeline
        
        Returns:
            Tuple containing generated video tensor in ComfyUI format
        """
        temp_audio_path = None
        
        try:
            # Ensure pipeline is loaded
            self._ensure_pipeline_loaded(config)
            
            # Validate inputs based on config
            if config.get('i2v', False) and image is None:
                warnings.warn("i2v mode enabled but no input image provided")
            
            if config.get('use_audio', True) and audio is None:
                warnings.warn("Audio processing enabled but no audio input provided")
            
            # Convert image to appropriate format
            processed_image = self._convert_image_to_tensor(image) if image is not None else None
            
            # Save audio to temporary file if provided
            if audio is not None:
                temp_audio_path = self._save_audio_to_temp_file(audio)
            
            # Prepare generation parameters
            generation_kwargs = {}
            if num_steps is not None:
                generation_kwargs['num_steps'] = num_steps
            if guidance_scale is not None:
                generation_kwargs['guidance_scale'] = guidance_scale
            if audio_scale is not None:
                generation_kwargs['audio_scale'] = audio_scale
            if negative_prompt:
                generation_kwargs['negative_prompt'] = negative_prompt
            if overlap_frame is not None:
                generation_kwargs['overlap_frame'] = overlap_frame
            
            print(f"[OmniAvatar] Generating video with prompt: '{prompt[:50]}...'")
            print(f"[OmniAvatar] Image input: {'Yes' if processed_image is not None else 'No'}")
            print(f"[OmniAvatar] Audio input: {'Yes' if temp_audio_path else 'No'}")
            print(f"[OmniAvatar] Dimensions: {width}x{height}")
            
            # Generate video
            with torch.cuda.amp.autocast(enabled=config.get('dtype') in ['fp16', 'bf16']):
                video = self.pipeline.forward(
                    prompt=prompt,
                    image_tensor=processed_image,
                    audio_path=temp_audio_path,
                    seq_len=seq_len,
                    height=height,
                    width=width,
                    **generation_kwargs
                )
            
            print(f"[OmniAvatar] Generated video shape: {video.shape}")
            
            # Convert to ComfyUI format
            comfyui_video = self._convert_video_to_comfyui_format(video)
            
            print(f"[OmniAvatar] Output video shape: {comfyui_video.shape}")
            print("[OmniAvatar] Video generation completed successfully")
            
            return (comfyui_video,)
            
        except Exception as e:
            error_msg = f"OmniAvatar generation failed: {str(e)}"
            print(f"[OmniAvatar ERROR] {error_msg}")
            
            # Create a simple error frame
            error_frame = torch.zeros((1, height, width, 3), dtype=torch.float32)
            return (error_frame,)
            
        finally:
            # Clean up temporary files
            self._cleanup_temp_files(temp_audio_path)


# Add utility functions for tensor format conversion
def comfyui_to_pytorch_image(image: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI image format to PyTorch format"""
    # ComfyUI: (H, W, C) or (B, H, W, C) in [0, 1]
    # PyTorch: (C, H, W) or (B, C, H, W) in [0, 1] or [-1, 1]
    
    if image.dim() == 3:  # (H, W, C)
        return image.permute(2, 0, 1)  # -> (C, H, W)
    elif image.dim() == 4:  # (B, H, W, C)
        return image.permute(0, 3, 1, 2)  # -> (B, C, H, W)
    else:
        return image


def pytorch_to_comfyui_image(image: torch.Tensor) -> torch.Tensor:
    """Convert PyTorch image format to ComfyUI format"""
    # PyTorch: (C, H, W) or (B, C, H, W) in [0, 1] or [-1, 1]
    # ComfyUI: (H, W, C) or (B, H, W, C) in [0, 1]
    
    # Normalize to [0, 1] if needed
    if image.min() < 0:
        image = (image + 1) / 2
    
    if image.dim() == 3:  # (C, H, W)
        return image.permute(1, 2, 0)  # -> (H, W, C)
    elif image.dim() == 4:  # (B, C, H, W)
        return image.permute(0, 2, 3, 1)  # -> (B, H, W, C)
    else:
        return image