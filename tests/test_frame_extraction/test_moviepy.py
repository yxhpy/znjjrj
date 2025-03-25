"""
MoviePy关键帧提取器测试
"""
import os
import json
import sys
import logging
import time
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
base_dir = current_dir.parent.parent
sys.path.insert(0, str(base_dir))

# 使用相对文件系统路径导入
sys.path.insert(0, str(current_dir))
from base_test import BaseFrameExtractorTest

from app.services.frame_extraction import (
    create_frame_extractor,
    MoviePyFrameExtractor
)


class TestMoviePyExtractor(BaseFrameExtractorTest):
    """MoviePy关键帧提取器测试类"""
    
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 设置环境变量
        os.environ['FRAME_EXTRACTION_STRATEGY'] = 'moviepy'
        os.environ['FRAME_EXTRACTION_CONFIG'] = json.dumps(self.test_config)
        
        # 确认视频文件存在
        if not os.path.exists(self.test_video_path):
            self.skipTest(f"测试视频文件不存在: {self.test_video_path}")
        
        # 输出视频文件信息
        file_size = os.path.getsize(self.test_video_path)
        logger.info(f"测试视频文件: {self.test_video_path}")
        logger.info(f"文件大小: {file_size} 字节")
        
        # 创建临时输出目录
        self.output_dir = os.path.join("output", f"test_output_{int(time.time())}")
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"测试输出目录: {self.output_dir}")
    
    def tearDown(self):
        """每个测试方法执行后的清理"""
        # 清理日志
        logger.info(f"测试完成: {self._testMethodName}")
    
    def test_extractor_creation(self):
        """测试MoviePy提取器创建"""
        logger.info("开始测试MoviePy提取器创建")
        extractor = create_frame_extractor()
        self.assertIsInstance(extractor, MoviePyFrameExtractor)
        logger.info("成功创建MoviePy提取器")
    
    def test_frame_extraction(self):
        """测试帧提取功能"""
        logger.info("开始测试MoviePy帧提取功能")
        
        # 创建提取器
        extractor = create_frame_extractor()
        
        try:
            # 获取视频信息
            info = extractor.get_video_info(self.test_video_path)
            logger.info(f"获取到视频信息: {info}")
            
            # 提取帧
            logger.info("开始提取帧")
            frames = extractor.extract_frames(self.test_video_path)
            logger.info(f"成功提取 {len(frames)} 帧")
            
            # 验证提取的帧
            self.assert_valid_frames(frames)
            
            # 保存第一帧用于查看
            if frames:
                first_frame = frames[0]
                import base64
                try:
                    frame_data = base64.b64decode(first_frame)
                    frame_path = os.path.join(self.output_dir, "test_frame.jpg")
                    with open(frame_path, 'wb') as f:
                        f.write(frame_data)
                    logger.info(f"保存测试帧到: {frame_path}")
                except Exception as e:
                    logger.error(f"保存测试帧失败: {str(e)}")
            
        except Exception as e:
            logger.error(f"测试帧提取失败: {str(e)}", exc_info=True)
            self.fail(f"帧提取测试失败: {str(e)}")


if __name__ == '__main__':
    import unittest
    unittest.main() 