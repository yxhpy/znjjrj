"""
FFmpeg关键帧提取器测试
"""
import os
import json
import tempfile
import sys
import logging
import shutil
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
    FFmpegFrameExtractor
)


class TestFFmpegExtractor(BaseFrameExtractorTest):
    """FFmpeg关键帧提取器测试类"""
    
    def setUp(self):
        """每个测试方法执行前的设置"""
        # 设置环境变量
        os.environ['FRAME_EXTRACTION_STRATEGY'] = 'ffmpeg'
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
        """测试FFmpeg提取器创建"""
        logger.info("开始测试FFmpeg提取器创建")
        extractor = create_frame_extractor()
        self.assertIsInstance(extractor, FFmpegFrameExtractor)
        logger.info("成功创建FFmpeg提取器")
    
    def test_frame_extraction(self):
        """测试帧提取功能"""
        logger.info("开始测试FFmpeg帧提取功能")
        
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
    
    def test_custom_config(self):
        """测试自定义配置"""
        logger.info("开始测试自定义配置")
        custom_config = {'max_frames': 3}
        try:
            extractor = create_frame_extractor('ffmpeg', custom_config)
            logger.info(f"使用自定义配置创建提取器: {custom_config}")
            frames = extractor.extract_frames(self.test_video_path)
            logger.info(f"成功提取 {len(frames)} 帧")
            self.assert_valid_frames(frames)
            self.assertLessEqual(len(frames), 3, "帧数应该不超过设置的最大帧数")
        except Exception as e:
            logger.error(f"测试自定义配置失败: {str(e)}", exc_info=True)
            self.fail(f"自定义配置测试失败: {str(e)}")
    
    def test_video_info(self):
        """测试视频信息获取"""
        logger.info("开始测试视频信息获取")
        try:
            extractor = create_frame_extractor('ffmpeg')
            info = extractor.get_video_info(self.test_video_path)
            logger.info(f"获取到视频信息: {info}")
            
            self.assertIn('width', info)
            self.assertIn('height', info)
            self.assertIn('duration', info)
            self.assertIn('fps', info)
            
            # 验证视频信息（根据实际视频调整这些值）
            self.assertGreater(info['width'], 0)
            self.assertGreater(info['height'], 0)
            self.assertGreater(info['duration'], 0)
            self.assertGreater(info['fps'], 0)
        except Exception as e:
            logger.error(f"测试视频信息获取失败: {str(e)}", exc_info=True)
            self.fail(f"视频信息获取测试失败: {str(e)}")
    
    def test_error_handling(self):
        """测试错误处理"""
        logger.info("开始测试错误处理")
        extractor = create_frame_extractor('ffmpeg')
        
        # 测试不存在的视频文件
        non_existent_file = 'nonexistent_' + str(int(time.time())) + '.mp4'
        logger.info(f"测试不存在的文件: {non_existent_file}")
        with self.assertRaises(Exception):
            extractor.extract_frames(non_existent_file)
        
        # 测试无效的视频文件
        logger.info("测试无效的视频文件")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            # 写入一些非视频数据
            temp_file.write(b'This is not a valid video file content')
            temp_file.flush()
            
            logger.info(f"创建无效视频文件: {temp_path}")
            
            try:
                # 尝试提取帧
                result = None
                try:
                    result = extractor.extract_frames(temp_path)
                except Exception as e:
                    logger.info(f"预期的异常: {str(e)}")
                
                # 如果没有抛出异常，确保没有提取到帧
                if result is not None:
                    self.assertEqual(len(result), 0, "无效视频文件不应该提取到任何帧")
            finally:
                # 清理临时文件
                try:
                    os.unlink(temp_path)
                    logger.info(f"删除临时文件: {temp_path}")
                except Exception as e:
                    logger.error(f"删除临时文件失败: {str(e)}")
                

if __name__ == '__main__':
    import unittest
    unittest.main() 