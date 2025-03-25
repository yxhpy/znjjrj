"""
关键帧提取器工厂类
"""
import json
import os
from typing import Dict, Any

from .base import FrameExtractor
from .ffmpeg_extractor import FFmpegFrameExtractor
from .moviepy_extractor import MoviePyFrameExtractor
from .opencv_extractor import OpenCVFrameExtractor
from .scene_detect_extractor import SceneDetectFrameExtractor

def create_frame_extractor(strategy: str = None, config: Dict[str, Any] = None) -> FrameExtractor:
    """
    创建关键帧提取器实例
    
    Args:
        strategy: 提取策略名称，如果为None则从环境变量获取
        config: 配置参数，如果为None则从环境变量获取
        
    Returns:
        FrameExtractor: 关键帧提取器实例
    """
    # 如果未指定策略，从环境变量获取
    if strategy is None:
        strategy = os.getenv('FRAME_EXTRACTION_STRATEGY', 'ffmpeg')
    
    # 如果未指定配置，从环境变量获取
    if config is None:
        config_str = os.getenv('FRAME_EXTRACTION_CONFIG', '{}')
        try:
            config = json.loads(config_str)
        except json.JSONDecodeError:
            config = {}
    
    # 策略映射
    strategies = {
        'ffmpeg': FFmpegFrameExtractor,
        'moviepy': MoviePyFrameExtractor,
        'opencv': OpenCVFrameExtractor,
        'scene_detect': SceneDetectFrameExtractor
    }
    
    # 获取提取器类
    extractor_class = strategies.get(strategy.lower())
    if not extractor_class:
        raise ValueError(f"不支持的提取策略: {strategy}")
    
    # 创建并返回提取器实例
    return extractor_class(config) 