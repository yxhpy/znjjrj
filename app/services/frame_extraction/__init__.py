"""
关键帧提取策略包
"""

from .base import FrameExtractor
from .ffmpeg_extractor import FFmpegFrameExtractor
from .moviepy_extractor import MoviePyFrameExtractor
from .opencv_extractor import OpenCVFrameExtractor
from .scene_detect_extractor import SceneDetectFrameExtractor
from .factory import create_frame_extractor

__all__ = [
    'FrameExtractor',
    'FFmpegFrameExtractor',
    'MoviePyFrameExtractor',
    'OpenCVFrameExtractor',
    'SceneDetectFrameExtractor',
    'create_frame_extractor',
] 