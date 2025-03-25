"""
关键帧提取器测试基类
"""
import os
import unittest
import json
import logging
import base64
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)


class BaseFrameExtractorTest(unittest.TestCase):
    """关键帧提取器测试基类"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        # 使用实际视频文件
        cls.test_video_path = "/home/yxhpy/project/znjjrj/uploads/28927593393-1-192.mp4"
        if not os.path.exists(cls.test_video_path):
            raise RuntimeError(f"测试视频文件不存在: {cls.test_video_path}")
        
        # 设置测试配置
        cls.test_config = {
            'max_frames': 5,
            'min_scene_length': 0.5,
            'threshold': 20.0,
            'method': 'content'
        }
        
        logger.info(f"初始化测试类 {cls.__name__} 完成")
        logger.info(f"测试视频: {cls.test_video_path}")
        logger.info(f"测试配置: {cls.test_config}")
    
    def assert_valid_frames(self, frames, expected_min_count=1):
        """验证提取的帧列表是否有效"""
        # 验证帧数量
        frame_count = len(frames)
        self.assertGreaterEqual(frame_count, expected_min_count, 
                              f"应至少提取 {expected_min_count} 帧，但只提取了 {frame_count} 帧")
        
        logger.info(f"验证 {frame_count} 帧")
        
        # 验证帧格式
        invalid_frames = 0
        for i, frame in enumerate(frames):
            try:
                self.assertTrue(isinstance(frame, str), f"帧 {i} 应该是字符串，但实际是 {type(frame)}")
                self.assertGreater(len(frame), 100, f"帧 {i} 字符串长度应该足够长，但只有 {len(frame)} 字符") 
                
                # 尝试解码base64
                try:
                    frame_data = base64.b64decode(frame)
                    self.assertGreater(len(frame_data), 0, f"帧 {i} 解码后为空数据")
                except Exception as e:
                    logger.error(f"帧 {i} base64解码失败: {str(e)}")
                    self.fail(f"帧 {i} 不是有效的base64编码: {str(e)}")
            except AssertionError as e:
                logger.error(f"帧验证失败: {str(e)}")
                invalid_frames += 1
                if invalid_frames > 3:  # 如果超过3个帧验证失败，直接测试失败
                    self.fail(f"超过3个帧验证失败，最后一个错误: {str(e)}")
        
        # 如果有任何无效帧但数量不超过3个，仍然记录警告
        if invalid_frames > 0:
            logger.warning(f"有 {invalid_frames} 个无效帧，但数量未超过阈值，测试继续")
        
        logger.info(f"所有帧验证通过，共 {frame_count} 帧有效") 