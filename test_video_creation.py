#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试视频创建功能

这个脚本用于测试视频服务的创建功能，特别是检验修复递归深度超限问题后的运行情况
"""

import os
import sys
import uuid
import logging
import tempfile
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("__main__")

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '')))

# 导入相关模块
from app.services.video_service import video_service
from app.schemas.video import VideoAnalysisResult, ChapterPoint
from app.core.database import VideoType

def create_test_video(duration=5.0):
    """创建测试视频文件"""
    import numpy as np
    from moviepy.editor import VideoClip
    
    # 创建测试视频文件夹
    temp_dir = Path("./temp")
    temp_dir.mkdir(exist_ok=True)
    
    # 生成唯一文件名
    test_id = uuid.uuid4()
    video_path = temp_dir / f"test_video_{test_id}.mp4"
    
    # 创建简单的测试视频
    def make_frame(t):
        # 创建简单的彩色帧
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 随时间变化的颜色
        r = int(255 * (t / duration))
        g = int(255 * (1 - t / duration))
        b = 100
        
        # 填充颜色
        frame[:, :, 0] = r
        frame[:, :, 1] = g
        frame[:, :, 2] = b
        
        return frame
    
    # 创建视频并写入文件
    clip = VideoClip(make_frame, duration=duration)
    clip.fps = 24
    clip.write_videofile(str(video_path), codec="libx264")
    
    logger.info(f"创建测试视频成功: {video_path}")
    return str(video_path)

def create_test_script():
    """创建测试解说脚本"""
    return """这是第一行测试文本。
这是第二行测试文本。
这是第三行测试文本。"""

def main():
    """主测试函数"""
    try:
        # 创建测试视频
        video_path = create_test_video()
        
        # 创建测试分析结果
        test_script = create_test_script()
        analysis_result = VideoAnalysisResult(
            narration_script=test_script,
            video_type=VideoType.MOVIE,
            duration=5.0,
            transcript="这是测试转录文本",
            chapter_points=[
                ChapterPoint(title="开始", time=0.0),
                ChapterPoint(title="中间", time=2.5),
                ChapterPoint(title="结尾", time=4.5)
            ]
        )
        
        # 生成测试任务ID
        test_id = f"test_{uuid.uuid4()}"
        
        # 测试视频创建
        logger.info("开始测试视频创建...")
        result_path = video_service.create_video_with_narration(
            video_path, 
            analysis_result, 
            test_id
        )
        
        logger.info(f"视频创建测试成功! 结果文件: {result_path}")
        
        # 清理测试文件
        if os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"已删除测试视频文件: {video_path}")
        
        return 0
    except Exception as e:
        logger.error(f"视频创建测试失败: {str(e)}")
        
        # 确保清理测试文件
        if 'video_path' in locals() and os.path.exists(video_path):
            os.remove(video_path)
            logger.info(f"已删除测试视频文件: {video_path}")
        
        logger.error("视频创建测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 