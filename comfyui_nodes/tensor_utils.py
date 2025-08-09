"""
Tensor conversion utilities for OmniAvatar ComfyUI nodes.
Handles format conversions between ComfyUI and PyTorch tensor formats.
"""

import torch
import numpy as np
from typing import Union, Optional, Tuple, Dict, Any
import warnings


def comfyui_to_pytorch_image(image: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI image format to PyTorch format.
    
    ComfyUI format: (H, W, C) or (B, H, W, C) in [0, 1] range
    PyTorch format: (C, H, W) or (B, C, H, W) in [0, 1] range
    
    Args:
        image: Input tensor in ComfyUI format
        
    Returns:
        Tensor in PyTorch format
    """
    if image is None:
        return None
    
    if image.dim() == 3:  # (H, W, C) -> (C, H, W)
        return image.permute(2, 0, 1)
    elif image.dim() == 4:  # (B, H, W, C) -> (B, C, H, W)
        return image.permute(0, 3, 1, 2)
    else:
        return image


def pytorch_to_comfyui_image(image: torch.Tensor) -> torch.Tensor:
    """
    Convert PyTorch image format to ComfyUI format.
    
    PyTorch format: (C, H, W) or (B, C, H, W) in [0, 1] or [-1, 1] range
    ComfyUI format: (H, W, C) or (B, H, W, C) in [0, 1] range
    
    Args:
        image: Input tensor in PyTorch format
        
    Returns:
        Tensor in ComfyUI format
    """
    if image is None:
        return None
    
    # Normalize to [0, 1] range if needed
    if image.min() < 0:
        image = torch.clamp((image + 1) / 2, 0, 1)
    else:
        image = torch.clamp(image, 0, 1)
    
    if image.dim() == 3:  # (C, H, W) -> (H, W, C)
        return image.permute(1, 2, 0)
    elif image.dim() == 4:  # (B, C, H, W) -> (B, H, W, C)
        return image.permute(0, 2, 3, 1)
    else:
        return image


def comfyui_to_pytorch_video(video: torch.Tensor) -> torch.Tensor:
    """
    Convert ComfyUI video format to PyTorch format.
    
    ComfyUI format: (T, H, W, C) in [0, 1] range
    PyTorch format: (1, T, C, H, W) or (T, C, H, W) in [0, 1] range
    
    Args:
        video: Input video tensor in ComfyUI format
        
    Returns:
        Tensor in PyTorch format
    """
    if video is None:
        return None
    
    if video.dim() == 4:  # (T, H, W, C) -> (1, T, C, H, W)
        video = video.permute(0, 3, 1, 2)  # -> (T, C, H, W)
        video = video.unsqueeze(0)  # -> (1, T, C, H, W)
    elif video.dim() == 5:  # (B, T, H, W, C) -> (B, T, C, H, W)
        video = video.permute(0, 1, 4, 2, 3)
    
    return video


def pytorch_to_comfyui_video(video: torch.Tensor) -> torch.Tensor:
    """
    Convert PyTorch video format to ComfyUI format.
    
    PyTorch format: (B, T, C, H, W) or (T, C, H, W) in [0, 1] or [-1, 1] range
    ComfyUI format: (T, H, W, C) in [0, 1] range
    
    Args:
        video: Input video tensor in PyTorch format
        
    Returns:
        Tensor in ComfyUI format
    """
    if video is None:
        return None
    
    # Normalize to [0, 1] range if needed
    if video.min() < 0:
        video = torch.clamp((video + 1) / 2, 0, 1)
    else:
        video = torch.clamp(video, 0, 1)
    
    if video.dim() == 5:  # (B, T, C, H, W) -> (T, H, W, C)
        video = video[0]  # Remove batch dimension -> (T, C, H, W)
    
    if video.dim() == 4:  # (T, C, H, W) -> (T, H, W, C)
        video = video.permute(0, 2, 3, 1)
    
    return video


def normalize_audio_format(audio: Union[torch.Tensor, Dict[str, Any], np.ndarray]) -> Optional[Dict[str, Any]]:
    """
    Normalize audio input to standard format.
    
    Args:
        audio: Audio input in various formats
        
    Returns:
        Dictionary with 'waveform' and 'sample_rate' keys, or None if invalid
    """
    if audio is None:
        return None
    
    # Handle dictionary format (ComfyUI standard)
    if isinstance(audio, dict):
        if 'waveform' in audio and 'sample_rate' in audio:
            return {
                'waveform': ensure_tensor(audio['waveform']),
                'sample_rate': int(audio['sample_rate'])
            }
        elif 'audio' in audio and 'sample_rate' in audio:
            return {
                'waveform': ensure_tensor(audio['audio']),
                'sample_rate': int(audio['sample_rate'])
            }
    
    # Handle raw tensor/array
    if isinstance(audio, (torch.Tensor, np.ndarray)):
        return {
            'waveform': ensure_tensor(audio),
            'sample_rate': 16000  # Default sample rate
        }
    
    warnings.warn(f"Unsupported audio format: {type(audio)}")
    return None


def ensure_tensor(data: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    """
    Ensure data is a PyTorch tensor.
    
    Args:
        data: Input data as tensor or array
        
    Returns:
        PyTorch tensor
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        return torch.tensor(data)


def ensure_rgb_channels(image: torch.Tensor) -> torch.Tensor:
    """
    Ensure image tensor has 3 RGB channels.
    
    Args:
        image: Input image tensor
        
    Returns:
        RGB image tensor
    """
    if image is None:
        return None
    
    # Handle different channel formats
    if image.dim() == 3:  # (C, H, W)
        if image.shape[0] == 1:  # Grayscale -> RGB
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:  # RGBA -> RGB
            image = image[:3]
    elif image.dim() == 4:  # (B, C, H, W)
        if image.shape[1] == 1:  # Grayscale -> RGB
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] == 4:  # RGBA -> RGB
            image = image[:, :3]
    
    return image


def match_tensor_device_dtype(
    tensor: torch.Tensor, 
    reference: torch.Tensor
) -> torch.Tensor:
    """
    Match tensor device and dtype to reference tensor.
    
    Args:
        tensor: Tensor to modify
        reference: Reference tensor for device/dtype
        
    Returns:
        Tensor with matched device and dtype
    """
    if tensor is None or reference is None:
        return tensor
    
    return tensor.to(device=reference.device, dtype=reference.dtype)


def safe_tensor_conversion(
    tensor: torch.Tensor,
    target_shape: Optional[Tuple[int, ...]] = None,
    target_dtype: Optional[torch.dtype] = None,
    target_device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Safely convert tensor with error handling.
    
    Args:
        tensor: Input tensor
        target_shape: Target shape (None to keep original)
        target_dtype: Target dtype (None to keep original)
        target_device: Target device (None to keep original)
        
    Returns:
        Converted tensor
    """
    if tensor is None:
        return None
    
    try:
        # Convert dtype if specified
        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        
        # Move to device if specified
        if target_device is not None and tensor.device != target_device:
            tensor = tensor.to(device=target_device)
        
        # Reshape if needed
        if target_shape is not None and tensor.shape != target_shape:
            # Only reshape if total elements match
            if tensor.numel() == np.prod(target_shape):
                tensor = tensor.reshape(target_shape)
            else:
                warnings.warn(f"Cannot reshape tensor from {tensor.shape} to {target_shape}")
        
        return tensor
        
    except Exception as e:
        warnings.warn(f"Tensor conversion failed: {e}")
        return tensor


def prepare_image_for_pipeline(
    image: torch.Tensor,
    target_size: Optional[Tuple[int, int]] = None,
    normalize: bool = True
) -> torch.Tensor:
    """
    Prepare image tensor for pipeline input.
    
    Args:
        image: Input image tensor
        target_size: Target (height, width) or None
        normalize: Whether to normalize to [-1, 1] range
        
    Returns:
        Prepared image tensor
    """
    if image is None:
        return None
    
    # Convert to PyTorch format if needed
    if image.shape[-1] in [1, 3, 4]:  # ComfyUI format (H, W, C)
        image = comfyui_to_pytorch_image(image)
    
    # Ensure RGB channels
    image = ensure_rgb_channels(image)
    
    # Resize if target size specified
    if target_size is not None:
        import torch.nn.functional as F
        image = F.interpolate(
            image.unsqueeze(0), 
            size=target_size, 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    # Normalize to [-1, 1] if requested
    if normalize and image.max() <= 1.0:
        image = image * 2.0 - 1.0
    
    return image


def prepare_video_for_output(
    video: torch.Tensor,
    fps: int = 25,
    format_type: str = 'comfyui'
) -> torch.Tensor:
    """
    Prepare video tensor for output.
    
    Args:
        video: Input video tensor
        fps: Frames per second
        format_type: Output format ('comfyui' or 'pytorch')
        
    Returns:
        Prepared video tensor
    """
    if video is None:
        return None
    
    # Clamp values to valid range
    video = torch.clamp(video, 0, 1)
    
    # Convert to requested format
    if format_type == 'comfyui':
        video = pytorch_to_comfyui_video(video)
    elif format_type == 'pytorch':
        video = comfyui_to_pytorch_video(video)
    
    return video


# Validation functions
def validate_image_tensor(tensor: torch.Tensor, name: str = "image") -> bool:
    """Validate image tensor format and values"""
    if tensor is None:
        return True
    
    if not isinstance(tensor, torch.Tensor):
        warnings.warn(f"{name} is not a torch.Tensor")
        return False
    
    if tensor.dim() not in [3, 4]:
        warnings.warn(f"{name} should have 3 or 4 dimensions, got {tensor.dim()}")
        return False
    
    if tensor.min() < 0 or tensor.max() > 1:
        warnings.warn(f"{name} values should be in [0, 1] range, got [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    return True


def validate_video_tensor(tensor: torch.Tensor, name: str = "video") -> bool:
    """Validate video tensor format and values"""
    if tensor is None:
        return True
    
    if not isinstance(tensor, torch.Tensor):
        warnings.warn(f"{name} is not a torch.Tensor")
        return False
    
    if tensor.dim() not in [4, 5]:
        warnings.warn(f"{name} should have 4 or 5 dimensions, got {tensor.dim()}")
        return False
    
    if tensor.min() < 0 or tensor.max() > 1:
        warnings.warn(f"{name} values should be in [0, 1] range, got [{tensor.min():.3f}, {tensor.max():.3f}]")
    
    return True


def validate_audio_dict(audio: Dict[str, Any], name: str = "audio") -> bool:
    """Validate audio dictionary format"""
    if audio is None:
        return True
    
    if not isinstance(audio, dict):
        warnings.warn(f"{name} should be a dictionary")
        return False
    
    if 'waveform' not in audio:
        warnings.warn(f"{name} dictionary missing 'waveform' key")
        return False
    
    if 'sample_rate' not in audio:
        warnings.warn(f"{name} dictionary missing 'sample_rate' key")
        return False
    
    waveform = audio['waveform']
    if not isinstance(waveform, torch.Tensor):
        warnings.warn(f"{name} waveform should be a torch.Tensor")
        return False
    
    return True