"""
使用PySceneDetect的关键帧提取器
"""
import os
import tempfile
from typing import List
import logging
from PIL import Image
from io import BytesIO

import cv2
import numpy as np
from scenedetect import detect, ContentDetector, AdaptiveDetector
from scenedetect import SceneManager
from scenedetect.video_manager import VideoManager

from .base import FrameExtractor

logger = logging.getLogger(__name__)

class SceneDetectFrameExtractor(FrameExtractor):
    """使用PySceneDetect提取关键帧的实现"""
    
    def __init__(self, config=None):
        """
        初始化场景检测提取器
        
        Args:
            config: 配置参数字典，可包含：
                - method: 检测方法 ('content' 或 'adaptive')
                - threshold: 检测阈值
                - min_scene_length: 最小场景长度(秒)
                - max_frames: 最大提取帧数
        """
        super().__init__(config)
        self.method = self.config.get('method', 'content')
        self.threshold = float(self.config.get('threshold', 30.0))
        self.min_scene_length = float(self.config.get('min_scene_length', 1.0))
    
    def extract_frames(self, video_path: str) -> List[str]:
        """
        使用场景检测从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[str]: base64编码的帧图像列表
        """
        try:
            # 创建视频管理器
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            
            # 获取fps
            video_manager.start()
            fps = video_manager.get_framerate()
            
            # 选择检测器
            if self.method.lower() == 'adaptive':
                detector = AdaptiveDetector()
            else:
                detector = ContentDetector(
                    threshold=self.threshold,
                    min_scene_len=int(self.min_scene_length * fps)
                )
            
            scene_manager.add_detector(detector)
            
            # 检测场景
            scene_manager.detect_scenes(video_manager)
            scenes = scene_manager.get_scene_list()
            logger.info(f"检测到 {len(scenes)} 个场景")
            
            if not scenes:
                logger.warning("未检测到场景，尝试使用固定间隔提取帧")
                video_manager.release()
                return self._extract_frames_by_interval(video_path)
            
            # 如果场景数量超过最大帧数，选择最显著的场景
            if len(scenes) > self.max_frames:
                # 按场景长度排序
                scenes = sorted(scenes, key=lambda x: x[1].get_frames() - x[0].get_frames(), reverse=True)
                scenes = scenes[:self.max_frames]
            
            frames_base64 = []
            cap = cv2.VideoCapture(video_path)
            
            for start_time, end_time in scenes:
                try:
                    # 获取场景中间帧 - 修复FrameTimecode类型问题
                    start_frame = int(start_time.get_frames())
                    end_frame = int(end_time.get_frames())
                    middle_frame = start_frame + (end_frame - start_frame) // 2
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
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
                        logger.info(f"成功提取场景 {start_time}-{end_time} 的关键帧（帧号：{middle_frame}）")
                    else:
                        logger.error(f"读取场景 {start_time}-{end_time} 的帧失败")
                except Exception as scene_error:
                    logger.error(f"处理场景时出错: {str(scene_error)}")
                    continue
            
            # 释放资源
            cap.release()
            video_manager.release()
            
            # 如果没有成功提取任何帧，尝试提取首帧
            if not frames_base64:
                logger.warning("未能提取任何场景帧，尝试提取视频首帧")
                return self._extract_first_frame(video_path)
            
            logger.info(f"成功提取 {len(frames_base64)} 个场景帧")
            return frames_base64
            
        except Exception as e:
            logger.error(f"场景检测提取帧失败: {str(e)}")
            # 确保资源被释放
            if 'video_manager' in locals():
                video_manager.release()
            if 'cap' in locals():
                cap.release()
            raise
    
    def _extract_frames_by_interval(self, video_path: str) -> List[str]:
        """使用固定间隔提取帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
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
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
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
                    logger.info(f"成功提取帧 {frame_idx}")
            
            cap.release()
            logger.info(f"通过间隔方式提取了 {len(frames_base64)} 帧")
            return frames_base64
            
        except Exception as e:
            logger.error(f"固定间隔提取帧失败: {str(e)}")
            if 'cap' in locals():
                cap.release()
            raise
    
    def _extract_first_frame(self, video_path: str) -> List[str]:
        """提取视频首帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            
            if ret:
                # 转换颜色空间从BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 转换为PIL图像并保存为JPEG
                pil_image = Image.fromarray(frame_rgb)
                buffer = BytesIO()
                pil_image.save(buffer, format="JPEG", quality=95)
                img_str = self.encode_frame(buffer.getvalue())
                
                cap.release()
                logger.info("成功提取视频首帧")
                return [img_str]
            
            cap.release()
            logger.error("提取首帧失败，视频可能无法读取")
            return []
            
        except Exception as e:
            logger.error(f"提取首帧失败: {str(e)}")
            if 'cap' in locals():
                cap.release()
            return [] 