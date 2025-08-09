import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from functools import partial
from typing import Optional, Dict, Any, Tuple
import warnings

# Try to import OmniAvatar dependencies - graceful fallback if not available
try:
    from OmniAvatar.models.model_manager import ModelManager
    from OmniAvatar.wan_video import WanVideoPipeline
    from OmniAvatar.utils.io_utils import load_state_dict
    from peft import LoraConfig, inject_adapter_in_model
    OMNIAVATAR_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"OmniAvatar dependencies not fully available: {e}")
    OMNIAVATAR_AVAILABLE = False
    ModelManager = None
    WanVideoPipeline = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    warnings.warn("librosa not available - audio processing disabled")
    LIBROSA_AVAILABLE = False

try:
    from transformers import Wav2Vec2FeatureExtractor
    from OmniAvatar.models.wav2vec import Wav2VecModel
    AUDIO_MODELS_AVAILABLE = True
except ImportError:
    warnings.warn("Audio model dependencies not available")
    AUDIO_MODELS_AVAILABLE = False


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def match_size(image_sizes, h, w):
    """Match input dimensions to closest available size"""
    ratio_ = 9999
    size_ = 9999
    select_size = None
    
    for image_s in image_sizes:
        ratio_tmp = abs(image_s[0] / image_s[1] - h / w)
        size_tmp = abs(max(image_s) - max(w, h))
        
        if ratio_tmp < ratio_:
            ratio_ = ratio_tmp
            size_ = size_tmp
            select_size = image_s
        elif ratio_ == ratio_tmp and size_tmp < size_:
            size_ = size_tmp
            select_size = image_s
    
    return select_size


def resize_pad(image, ori_size, tgt_size):
    """Resize and pad image to target size"""
    h, w = ori_size
    scale_ratio = max(tgt_size[0] / h, tgt_size[1] / w)
    scale_h = int(h * scale_ratio)
    scale_w = int(w * scale_ratio)

    image = transforms.Resize(size=[scale_h, scale_w])(image)

    padding_h = tgt_size[0] - scale_h
    padding_w = tgt_size[1] - scale_w
    pad_top = padding_h // 2
    pad_bottom = padding_h - pad_top
    pad_left = padding_w // 2
    pad_right = padding_w - pad_left

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return image


class SimplifiedWanInferencePipeline(nn.Module):
    """
    Simplified WanInferencePipeline for ComfyUI compatibility
    Handles single GPU inference without distributed processing
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        if not OMNIAVATAR_AVAILABLE:
            raise ImportError(
                "OmniAvatar dependencies not available. Please install OmniAvatar package.\n"
                "Required modules: OmniAvatar.models.model_manager, OmniAvatar.wan_video"
            )
        
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set data type
        dtype_map = {
            'bf16': torch.bfloat16,
            'fp16': torch.float16,
            'fp32': torch.float32
        }
        self.dtype = dtype_map.get(config.get('dtype', 'bf16'), torch.bfloat16)
        
        # Initialize pipeline
        self.pipe = self.load_model()
        
        # Initialize image transforms for i2v mode
        if config.get('i2v', False):
            self.transform = transforms.Compose([transforms.ToTensor()])
        
        # Initialize audio components if needed
        self.audio_encoder = None
        self.wav_feature_extractor = None
        if config.get('use_audio', True) and AUDIO_MODELS_AVAILABLE:
            self._init_audio_components()
    
    def _init_audio_components(self):
        """Initialize audio processing components"""
        try:
            wav2vec_path = self.config.get('wav2vec_path', '')
            if wav2vec_path:
                self.wav_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path)
                self.audio_encoder = Wav2VecModel.from_pretrained(
                    wav2vec_path, 
                    local_files_only=True
                ).to(device=self.device)
                self.audio_encoder.feature_extractor._freeze_parameters()
        except Exception as e:
            warnings.warn(f"Failed to initialize audio components: {e}")
            self.audio_encoder = None
            self.wav_feature_extractor = None
    
    def load_model(self):
        """Load the model pipeline - simplified for single GPU"""
        try:
            # Check if checkpoint exists
            ckpt_path = f"{self.config['exp_path']}/pytorch_model.pt"
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"pytorch_model.pt not found in {self.config['exp_path']}")
            
            # Load models using ModelManager
            model_manager = ModelManager(device="cpu", infer=True)
            
            dit_paths = self.config['dit_path'].split(',')
            model_manager.load_models(
                [dit_paths, self.config['text_encoder_path'], self.config['vae_path']],
                torch_dtype=self.dtype,
                device='cpu',
            )
            
            # Create pipeline
            pipe = WanVideoPipeline.from_model_manager(
                model_manager,
                torch_dtype=self.dtype,
                device=self.device,
                use_usp=False,  # Simplified for single GPU
                infer=True
            )
            
            # Handle LoRA or regular model loading
            if self.config.get('train_architecture') == 'lora':
                self._add_lora_to_model(pipe.denoising_model(), ckpt_path)
            else:
                missing_keys, unexpected_keys = pipe.denoising_model().load_state_dict(
                    load_state_dict(ckpt_path), strict=True
                )
                print(f"Loaded from {ckpt_path}, {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
            
            pipe.requires_grad_(False)
            pipe.eval()
            
            # Enable VRAM management if specified
            num_persistent = self.config.get('num_persistent_param_in_dit')
            if num_persistent is not None:
                pipe.enable_vram_management(num_persistent_param_in_dit=num_persistent)
            
            return pipe
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _add_lora_to_model(self, model, pretrained_lora_path: str):
        """Add LoRA adapter to model"""
        try:
            lora_config = LoraConfig(
                r=self.config.get('lora_rank', 4),
                lora_alpha=self.config.get('lora_alpha', 4),
                init_lora_weights=self.config.get('init_lora_weights', 'kaiming') == 'kaiming',
                target_modules=self.config.get('lora_target_modules', 'q,k,v,o,ffn.0,ffn.2').split(','),
            )
            
            model = inject_adapter_in_model(lora_config, model)
            
            # Load pretrained LoRA weights
            state_dict = load_state_dict(pretrained_lora_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            all_keys = [name for name, _ in model.named_parameters()]
            num_updated = len(all_keys) - len(missing_keys)
            print(f"{num_updated} LoRA parameters loaded from {pretrained_lora_path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to add LoRA: {e}")
    
    def process_audio(self, audio_path: str, seq_len: int, first_fixed_frame: int = 0) -> Optional[torch.Tensor]:
        """Process audio file for audio-driven generation"""
        if not self.config.get('use_audio', True) or not LIBROSA_AVAILABLE or not self.audio_encoder:
            return None
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.config.get('sample_rate', 16000))
            
            # Extract features
            input_values = np.squeeze(
                self.wav_feature_extractor(audio, sampling_rate=16000).input_values
            )
            input_values = torch.from_numpy(input_values).float().to(device=self.device)
            
            # Calculate audio length in frames
            audio_len = math.ceil(len(input_values) / self.config['sample_rate'] * self.config['fps'])
            input_values = input_values.unsqueeze(0)
            
            # Pad audio to match video length
            target_len = audio_len * int(self.config['sample_rate'] / self.config['fps'])
            if input_values.shape[1] < target_len:
                input_values = F.pad(
                    input_values, 
                    (0, target_len - input_values.shape[1]), 
                    mode='constant', value=0
                )
            
            # Extract audio embeddings
            with torch.no_grad():
                hidden_states = self.audio_encoder(input_values, seq_len=audio_len, output_hidden_states=True)
                audio_embeddings = hidden_states.last_hidden_state
                
                # Concatenate hidden states
                for mid_hidden_states in hidden_states.hidden_states:
                    audio_embeddings = torch.cat((audio_embeddings, mid_hidden_states), -1)
            
            return audio_embeddings.squeeze(0)
            
        except Exception as e:
            warnings.warn(f"Failed to process audio: {e}")
            return None
    
    def process_image(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Process image for i2v generation"""
        if not self.config.get('i2v', False):
            return None, None
        
        try:
            # Convert to appropriate format and device
            if image_tensor.dim() == 4:  # Batch dimension
                image_tensor = image_tensor[0]
            
            image_tensor = image_tensor.to(self.device)
            _, h, w = image_tensor.shape
            
            # Select appropriate image size
            size_key = f"image_sizes_{self.config['max_hw']}"
            image_sizes = self.config.get(size_key, [[720, 720]])
            select_size = match_size(image_sizes, h, w)
            
            # Resize and pad
            image_tensor = resize_pad(image_tensor, (h, w), select_size)
            
            # Normalize to [-1, 1]
            image_tensor = image_tensor * 2.0 - 1.0
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(2)  # Add batch and temporal dimensions
            
            return image_tensor, tuple(select_size)
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {e}")
    
    def forward(
        self,
        prompt: str,
        image_tensor: Optional[torch.Tensor] = None,
        audio_path: Optional[str] = None,
        seq_len: Optional[int] = None,
        height: int = 720,
        width: int = 720,
        **generation_kwargs
    ) -> torch.Tensor:
        """
        Generate video from prompt with optional image and audio conditioning
        
        Args:
            prompt: Text prompt for generation
            image_tensor: Input image tensor for i2v mode (optional)
            audio_path: Path to audio file for audio-driven generation (optional)  
            seq_len: Sequence length override (optional)
            height: Video height (used when no image provided)
            width: Video width (used when no image provided)
            **generation_kwargs: Additional generation parameters
            
        Returns:
            Generated video tensor
        """
        try:
            # Set seed for reproducibility
            set_seed(self.config.get('seed', 42))
            
            # Get generation parameters with fallbacks
            seq_len = seq_len or self.config.get('seq_len', 200)
            num_steps = generation_kwargs.get('num_steps', self.config.get('num_steps', 50))
            guidance_scale = generation_kwargs.get('guidance_scale', self.config.get('guidance_scale', 4.5))
            audio_scale = generation_kwargs.get('audio_scale', self.config.get('audio_scale', guidance_scale))
            negative_prompt = generation_kwargs.get('negative_prompt', self.config.get('negative_prompt', ''))
            overlap_frame = generation_kwargs.get('overlap_frame', self.config.get('overlap_frame', 13))
            
            # Process image if provided
            image = None
            select_size = [height, width]
            if image_tensor is not None:
                image, select_size = self.process_image(image_tensor)
            
            # Calculate video dimensions
            L = int(self.config['max_tokens'] * 16 * 16 * 4 / select_size[0] / select_size[1])
            L = L // 4 * 4 + 1 if L % 4 != 0 else L - 3
            T = (L + 3) // 4
            
            # Process audio if provided
            audio_embeddings = None
            if audio_path:
                audio_embeddings = self.process_audio(audio_path, seq_len)
                if audio_embeddings is not None:
                    seq_len = min(seq_len, audio_embeddings.shape[0])
            
            # Setup for i2v mode
            fixed_frame = 0
            first_fixed_frame = 0
            if self.config.get('i2v', False) and image is not None:
                if self.config.get('random_prefix_frames', False):
                    fixed_frame = overlap_frame
                    assert fixed_frame % 4 == 1
                else:
                    fixed_frame = 0
                first_fixed_frame = 1
            
            # Generate video
            video_frames = []
            image_emb = {}
            img_lat = None
            
            # Encode image if in i2v mode
            if self.config.get('i2v', False) and image is not None:
                self.pipe.load_models_to_device(['vae'])
                img_lat = self.pipe.encode_video(image.to(dtype=self.dtype)).to(self.device)
                
                # Setup image embeddings
                msk = torch.zeros_like(img_lat.repeat(1, 1, T, 1, 1)[:, :1])
                image_cat = img_lat.repeat(1, 1, T, 1, 1)
                msk[:, :, 1:] = 1
                image_emb["y"] = torch.cat([image_cat, msk], dim=1)
            
            # Calculate number of generation loops
            times = (seq_len - L + first_fixed_frame) // (L - fixed_frame) + 1
            if times * (L - fixed_frame) + fixed_frame < seq_len:
                times += 1
            
            # Generate video in chunks
            for t in range(times):
                print(f"[{t+1}/{times}]")
                
                audio_emb = {}
                overlap = first_fixed_frame if t == 0 else fixed_frame
                prefix_overlap = (3 + overlap) // 4
                
                # Setup audio embeddings for this chunk
                if audio_embeddings is not None:
                    if t == 0:
                        audio_tensor = audio_embeddings[:min(L - overlap, audio_embeddings.shape[0])]
                    else:
                        audio_start = L - first_fixed_frame + (t - 1) * (L - overlap)
                        audio_tensor = audio_embeddings[
                            audio_start:min(audio_start + L - overlap, audio_embeddings.shape[0])
                        ]
                    
                    # Add audio prefix and convert to proper format
                    audio_prefix = torch.zeros_like(audio_embeddings[:first_fixed_frame])
                    audio_tensor = torch.cat([audio_prefix, audio_tensor], dim=0)
                    audio_tensor = audio_tensor.unsqueeze(0).to(device=self.device, dtype=self.dtype)
                    audio_emb["audio_emb"] = audio_tensor
                
                # Setup image latents for this chunk
                if img_lat is not None:
                    img_lat = torch.cat([
                        img_lat, 
                        torch.zeros_like(img_lat[:, :, :1].repeat(1, 1, T - prefix_overlap, 1, 1))
                    ], dim=2)
                
                # Generate frames for this chunk
                frames, _, latents = self.pipe.log_video(
                    img_lat, prompt, prefix_overlap, image_emb, audio_emb,
                    negative_prompt, 
                    num_inference_steps=num_steps,
                    cfg_scale=guidance_scale, 
                    audio_cfg_scale=audio_scale,
                    return_latent=True,
                    tea_cache_l1_thresh=self.config.get('tea_cache_l1_thresh', 0),
                    tea_cache_model_id="Wan2.1-T2V-14B"
                )
                
                # Process frames for next iteration
                img_lat = None
                if t == 0:
                    video_frames.append(frames)
                else:
                    video_frames.append(frames[:, overlap:])
                
                # Update image for next iteration
                if frames.shape[1] > fixed_frame:
                    image = (frames[:, -fixed_frame:].clip(0, 1) * 2 - 1).permute(0, 2, 1, 3, 4).contiguous()
            
            # Concatenate all video frames
            video = torch.cat(video_frames, dim=1)
            
            # Trim to original sequence length if audio was used
            if audio_embeddings is not None:
                orig_len = min(seq_len + 1, video.shape[1])
                video = video[:, :orig_len]
            
            return video
            
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")


# Add necessary imports at the top
import os