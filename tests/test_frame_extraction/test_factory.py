"""
关键帧提取器工厂测试
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent.parent
sys.path.insert(0, str(base_dir))

# 使用相对文件系统路径导入
base_test_path = os.path.join(current_dir, "base_test.py")
if not os.path.exists(base_test_path):
    raise FileNotFoundError(f"找不到基础测试模块: {base_test_path}")

sys.path.insert(0, str(current_dir))
from base_test import BaseFrameExtractorTest

from app.services.frame_extraction import (
    create_frame_extractor,
    FFmpegFrameExtractor,
    MoviePyFrameExtractor,
    OpenCVFrameExtractor,
    SceneDetectFrameExtractor
)


class TestFrameExtractorFactory(BaseFrameExtractorTest):
    """关键帧提取器工厂测试类"""
    
    def test_create_ffmpeg_extractor(self):
        """测试创建FFmpeg提取器"""
        extractor = create_frame_extractor('ffmpeg')
        self.assertIsInstance(extractor, FFmpegFrameExtractor)
    
    def test_create_moviepy_extractor(self):
        """测试创建MoviePy提取器"""
        extractor = create_frame_extractor('moviepy')
        self.assertIsInstance(extractor, MoviePyFrameExtractor)
    
    def test_create_opencv_extractor(self):
        """测试创建OpenCV提取器"""
        extractor = create_frame_extractor('opencv')
        self.assertIsInstance(extractor, OpenCVFrameExtractor)
    
    def test_create_scene_detect_extractor(self):
        """测试创建场景检测提取器"""
        extractor = create_frame_extractor('scene_detect')
        self.assertIsInstance(extractor, SceneDetectFrameExtractor)
    
    def test_invalid_strategy(self):
        """测试无效的提取策略"""
        with self.assertRaises(ValueError):
            create_frame_extractor('invalid_strategy')
    
    def test_custom_config(self):
        """测试自定义配置"""
        custom_config = {'max_frames': 3}
        extractor = create_frame_extractor('ffmpeg', custom_config)
        self.assertEqual(extractor.config['max_frames'], 3)


if __name__ == '__main__':
    import unittest
    unittest.main() 