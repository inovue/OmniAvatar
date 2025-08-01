# OmniAvatar: Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation
# Authors: Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi
# Zhejiang University, Alibaba Group

__version__ = "0.1.0"
__description__ = "Efficient Audio-Driven Avatar Video Generation with Adaptive Body Animation"
__author__ = "Qijun Gan, Ruizi Yang, Jianke Zhu, Shaofei Xue, Steven Hoi"
__email__ = "ganqijun@zju.edu.cn"
__url__ = "https://omni-avatar.github.io/"

from .wan_video import WanVideoPipeline, TeaCache
from .base import BasePipeline
from .models.model_manager import ModelManager
from .prompters import WanPrompter

# Main pipeline class
__all__ = [
    "WanVideoPipeline",
    "TeaCache", 
    "BasePipeline",
    "ModelManager",
    "WanPrompter",
    "__version__",
    "__description__",
    "__author__",
    "__email__",
    "__url__",
]