"""
使用OpenCV的关键帧提取器
"""
import cv2
import numpy as np
from typing import List
import logging
from PIL import Image
from io import BytesIO

from .base import FrameExtractor

logger = logging.getLogger(__name__)

class OpenCVFrameExtractor(FrameExtractor):
    """使用OpenCV提取关键帧的实现"""
    
    def extract_frames(self, video_path: str) -> List[str]:
        """
        使用OpenCV从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[str]: base64编码的帧图像列表
        """
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"无法打开视频文件: {video_path}")
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"视频信息: 总帧数={total_frames}, FPS={fps}, 时长={duration:.2f}秒")
            
            # 计算帧间隔
            if total_frames <= self.max_frames:
                frame_indices = range(total_frames)
            else:
                frame_indices = [
                    int(i * total_frames / (self.max_frames + 1))
                    for i in range(1, self.max_frames + 1)
                ]
            
            frames_base64 = []
            for frame_idx in frame_indices:
                # 设置当前帧位置
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    try:
                        # 转换颜色空间从BGR到RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # 转换为PIL图像并保存为JPEG
                        pil_image = Image.fromarray(frame_rgb)
                        buffer = BytesIO()
                        pil_image.save(buffer, format="JPEG", quality=95)
                        img_str = self.encode_frame(buffer.getvalue())
                        
                        frames_base64.append(img_str)
                        logger.info(f"成功提取帧 {frame_idx}")
                    except Exception as frame_error:
                        logger.error(f"处理帧 {frame_idx} 时出错: {str(frame_error)}")
                        continue
                else:
                    logger.error(f"读取帧 {frame_idx} 失败")
            
            # 如果没有成功提取任何帧，尝试提取首帧
            if not frames_base64:
                logger.warning("未能提取任何帧，尝试提取视频首帧")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
                
                if ret:
                    # 转换颜色空间从BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # 转换为PIL图像并保存为JPEG
                    pil_image = Image.fromarray(frame_rgb)
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG", quality=95)
                    img_str = self.encode_frame(buffer.getvalue())
                    
                    frames_base64.append(img_str)
            
            # 释放视频对象
            cap.release()
            
            logger.info(f"成功提取 {len(frames_base64)} 帧")
            return frames_base64
            
        except Exception as e:
            logger.error(f"OpenCV提取帧失败: {str(e)}")
            # 确保视频对象被释放
            if 'cap' in locals():
                cap.release()
            raise 