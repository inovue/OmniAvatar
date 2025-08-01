# OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation
# Authors: Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi
# Zhejiang University, Alibaba Group

__version__ = "0.1.0"
__description__ = "Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation"
__author__ = "Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi"
__email__ = "ganqijun@zju.edu.cn"
__url__ = "https://omni-avatar.github.io/"

# Core modules
from .wan_video import WanVideoPipeline, TeaCache
from .base import BasePipeline

# Models
from .models.model_manager import ModelManager
from .models.audio_pack import AudioPack
from .models.wan_video_dit import WanModel
from .models.wan_video_text_encoder import WanTextEncoder
from .models.wan_video_vae import WanVideoVAE
from .models.wav2vec import Wav2VecModel

# Prompters
from .prompters import WanPrompter

# Configs
from .configs.model_config import *

# Distributed
from .distributed.fsdp import *
from .distributed.xdit_context_parallel import *

# Schedulers
from .schedulers.flow_match import FlowMatchScheduler

# Utils
from .utils.args_config import parse_args, parse_hp_string, reload, convert_namespace_to_dict
from .utils.audio_preprocess import add_silence_to_audio_ffmpeg
from .utils.io_utils import (
    load_state_dict, save_wav, save_video_as_grid_and_mp4,
    hash_state_dict_keys, split_state_dict_with_prefix
)

# VRAM Management
from .vram_management import *

# Main pipeline class
__all__ = [
    # Core
    "WanVideoPipeline",
    "TeaCache", 
    "BasePipeline",
    
    # Models
    "ModelManager",
    "AudioPack",
    "WanModel",
    "WanTextEncoder",
    "WanVideoVAE", 
    "Wav2VecModel",
    
    # Prompters
    "WanPrompter",
    
    # Schedulers
    "FlowMatchScheduler",
    
    # Utils
    "parse_args",
    "parse_hp_string",
    "reload",
    "convert_namespace_to_dict",
    "add_silence_to_audio_ffmpeg",
    "load_state_dict",
    "save_wav",
    "save_video_as_grid_and_mp4",
    "hash_state_dict_keys",
    "split_state_dict_with_prefix",
    
    # Metadata
    "__version__",
    "__description__",
    "__author__",
    "__email__",
    "__url__",
]