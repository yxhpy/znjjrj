"""
使用MoviePy的关键帧提取器
"""
import os
import tempfile
from typing import List
import logging
import numpy as np
from PIL import Image
from io import BytesIO
from moviepy.editor import VideoFileClip

from .base import FrameExtractor

logger = logging.getLogger(__name__)

class MoviePyFrameExtractor(FrameExtractor):
    """使用MoviePy提取关键帧的实现"""
    
    def extract_frames(self, video_path: str) -> List[str]:
        """
        使用MoviePy从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[str]: base64编码的帧图像列表
        """
        try:
            # 加载视频
            clip = VideoFileClip(video_path)
            duration = clip.duration
            logger.info(f"视频时长: {duration}秒, 分辨率: {clip.size}")
            
            # 计算提取帧的时间点
            timestamps = [i * duration / (self.max_frames + 1) for i in range(1, self.max_frames + 1)]
            
            # 提取帧并编码
            frames_base64 = []
            for ts in timestamps:
                try:
                    # 获取指定时间的帧
                    frame = clip.get_frame(ts)
                    logger.info(f"帧形状: {frame.shape}, 帧类型: {frame.dtype}")
                    
                    # 检查帧的形状并适当处理
                    if len(frame.shape) == 3:
                        if frame.shape[0] == 3:  # 通道在第一维 (3, H, W)
                            logger.info("帧格式为 (通道,高,宽)，进行转置")
                            frame = np.transpose(frame, (1, 2, 0))
                        elif frame.shape[2] != 3:  # 非标准RGB格式
                            logger.info(f"非标准RGB格式，形状为 {frame.shape}，尝试转换")
                            if frame.shape[2] == 4:  # RGBA格式
                                frame = frame[:, :, :3]  # 去掉Alpha通道
                    else:
                        logger.error(f"无法处理的帧形状: {frame.shape}")
                        continue
                    
                    # 确保数值范围在0-255之间
                    if frame.dtype != np.uint8:
                        logger.info(f"转换帧数据类型从 {frame.dtype} 到 uint8")
                        if frame.max() <= 1.0:  # 如果值在0-1之间
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    # 转换为PIL图像并保存为JPEG
                    pil_image = Image.fromarray(frame)
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=95)
                    img_str = self.encode_frame(buffer.getvalue())
                    
                    frames_base64.append(img_str)
                    logger.info(f"成功提取时间点 {ts:.2f}s 的帧")
                except Exception as frame_error:
                    logger.error(f"处理时间点 {ts:.2f}s 的帧时出错: {str(frame_error)}")
                    continue
            
            # 关闭视频
            clip.close()
            
            # 如果没有成功提取任何帧，尝试提取首帧
            if not frames_base64:
                logger.warning("未能提取任何帧，尝试提取视频首帧")
                try:
                    clip = VideoFileClip(video_path)
                    frame = clip.get_frame(0)
                    
                    # 转换为PIL图像并保存为JPEG
                    pil_image = Image.fromarray(frame.astype(np.uint8))
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=95)
                    img_str = self.encode_frame(buffer.getvalue())
                    
                    frames_base64.append(img_str)
                    clip.close()
                except Exception as e:
                    logger.error(f"提取首帧失败: {str(e)}")
                    raise
            
            logger.info(f"成功提取 {len(frames_base64)} 帧")
            return frames_base64
            
        except Exception as e:
            logger.error(f"MoviePy提取帧失败: {str(e)}")
            raise 