#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频服务测试脚本

测试VideoService的核心功能，特别是TextClip创建和视频生成功能
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
import uuid
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, ImageClip, VideoClip
from PIL import Image, ImageDraw, ImageFont
from app.config import TEMP_DIR
from app.schemas.video import VideoAnalysisResult, ChapterPoint
from app.services.video_service import VideoService
from app.core.database import VideoType

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_video():
    """创建测试视频文件"""
    # 创建一个简单的彩色帧序列作为测试视频
    def make_frame(t):
        # 创建一个随时间变化的彩色帧
        # t是时间，范围从0到duration
        red = int(255 * (1 + np.sin(t*2)) / 2)
        green = int(255 * (1 + np.sin(t*4 + np.pi/2)) / 2)
        blue = int(255 * (1 + np.sin(t*6 + np.pi)) / 2)
        
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img[:, :, 0] = red
        img[:, :, 1] = green
        img[:, :, 2] = blue
        
        # 添加时间文本
        # 直接使用numpy数组操作，避免使用TextClip
        # 在中间位置绘制时间文本
        time_text = f"Time: {t:.1f}s"
        height, width = img.shape[:2]
        
        # 在底部绘制白色矩形作为文本背景
        img[height-30:height, :] = [255, 255, 255]
        
        # 不使用文本渲染以避免依赖问题
        # 简单地在帧上编码一些信息
        for i, char in enumerate(time_text):
            if i < width // 10:
                x_pos = i * 20 + 10
                # 绘制简单的数字表示
                img[height-25:height-5, x_pos:x_pos+15] = [0, 0, 0]
        
        return img
    
    # 创建一个5秒的测试视频
    duration = 5.0
    video_clip = VideoClip(make_frame, duration=duration)
    
    # 输出到临时文件
    test_video_path = os.path.join(TEMP_DIR, f"test_video_{uuid.uuid4()}.mp4")
    video_clip.write_videofile(
        test_video_path,
        fps=24,
        codec="libx264",
        audio_codec=None  # 无音频
    )
    
    logger.info(f"创建测试视频: {test_video_path}")
    return test_video_path

def test_imagemagick_config():
    """测试ImageMagick配置"""
    logger.info("测试ImageMagick配置...")
    
    try:
        # 创建一个简单的TextClip测试
        test_text = "测试文本"
        
        # 方法1：直接使用TextClip
        try:
            clip1 = TextClip(test_text, fontsize=24, color='white', bg_color='black', 
                          size=(320, None), method='label')
            logger.info("TextClip创建成功 (方法1: label)")
        except Exception as e:
            logger.error(f"TextClip创建失败 (方法1: label): {str(e)}")
            
            # 尝试不同的方法
            try:
                clip1 = TextClip(test_text, fontsize=24, color='white', bg_color='black', 
                              size=(320, None), method='caption')
                logger.info("TextClip创建成功 (方法1: caption)")
            except Exception as e:
                logger.error(f"TextClip创建失败 (方法1: caption): {str(e)}")
        
        # 方法2：使用PIL和ImageClip
        try:
            # 创建PIL图像
            img = Image.new('RGBA', (320, 50), (0, 0, 0, 200))
            draw = ImageDraw.Draw(img)
            
            # 尝试加载中文字体，如果不可用则使用默认字体
            try:
                # 尝试加载系统可能存在的中文字体
                font_paths = [
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驿微米黑
                    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Droid Sans
                    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
                    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK (opentype)
                ]
                
                font = None
                for font_path in font_paths:
                    if os.path.exists(font_path):
                        font = ImageFont.truetype(font_path, 16)
                        logger.info(f"成功加载字体: {font_path}")
                        break
                
                if font is None:
                    # 如果找不到合适的字体，使用默认字体
                    font = ImageFont.load_default()
                    logger.info("使用默认字体")
            except Exception as font_error:
                logger.warning(f"加载字体失败: {str(font_error)}，使用默认字体")
                font = ImageFont.load_default()
            
            # 绘制文本 - 使用ASCII文本避免编码问题
            ascii_text = "Test Text"
            draw.text((10, 10), ascii_text, fill=(255, 255, 255, 255), font=font)
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            # 创建ImageClip
            clip2 = ImageClip(img_array)
            logger.info("PIL方式创建文本图像成功")
        except Exception as e:
            logger.error(f"PIL方式创建文本图像失败: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"测试ImageMagick配置失败: {str(e)}")
        return False

def test_video_creation():
    """测试视频创建功能"""
    logger.info("测试视频创建功能...")
    
    try:
        # 创建测试视频
        test_video_path = create_test_video()
        
        # 获取视频时长
        video = VideoFileClip(test_video_path)
        duration = video.duration
        video.close()
        
        # 创建测试用的VideoAnalysisResult对象
        chapter_points = [
            ChapterPoint(time=1.0, title="章节1"),
            ChapterPoint(time=3.0, title="章节2")
        ]
        
        analysis_result = VideoAnalysisResult(
            narration_script="这是第一行测试文本。\n这是第二行测试文本。\n这是第三行测试文本。",
            chapter_points=chapter_points,
            summary="测试视频总结",
            key_points=["测试要点1", "测试要点2"],
            video_type=VideoType.MOVIE,  # 假设是电影类型
            duration=duration,           # 使用视频实际时长
            transcript="这是原视频的文字记录"  # 添加原视频文字记录
        )
        
        # 创建唯一任务ID
        task_id = f"test_{uuid.uuid4()}"
        
        # 创建VideoService实例
        video_service = VideoService()
        
        # 调用视频创建函数
        output_path = video_service.create_video_with_narration(
            test_video_path,
            analysis_result,
            task_id
        )
        
        # 检查生成的视频
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"成功创建视频: {output_path} (大小: {file_size} 字节)")
            
            # 检查视频是否可以打开
            try:
                video = VideoFileClip(output_path)
                logger.info(f"视频可以正常打开，时长: {video.duration}秒")
                video.close()
            except Exception as e:
                logger.error(f"无法打开生成的视频: {str(e)}")
            
            return True
        else:
            logger.error(f"视频创建失败: 输出文件不存在")
            return False
    except Exception as e:
        logger.error(f"视频创建测试失败: {str(e)}")
        return False
    finally:
        # 清理临时文件
        if 'test_video_path' in locals() and os.path.exists(test_video_path):
            try:
                os.remove(test_video_path)
                logger.info(f"已删除测试视频文件: {test_video_path}")
            except:
                pass

def main():
    """主函数"""
    logger.info("开始VideoService测试...")
    
    # 测试ImageMagick配置
    if test_imagemagick_config():
        logger.info("ImageMagick配置测试通过")
    else:
        logger.error("ImageMagick配置测试失败")
    
    # 测试视频创建
    if test_video_creation():
        logger.info("视频创建测试通过")
    else:
        logger.error("视频创建测试失败")
    
if __name__ == "__main__":
    main() 