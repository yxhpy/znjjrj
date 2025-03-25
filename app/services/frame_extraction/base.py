"""
关键帧提取器基类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import base64
import logging
import os

logger = logging.getLogger(__name__)

class FrameExtractor(ABC):
    """关键帧提取器基类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化提取器
        
        Args:
            config: 配置参数字典
        """
        self.config = config or {}
        self.max_frames = self.config.get('max_frames', 10)
    
    @abstractmethod
    def extract_frames(self, video_path: str) -> List[str]:
        """
        从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[str]: base64编码的帧图像列表
        """
        pass
    
    def encode_frame(self, frame_data: bytes) -> str:
        """
        将帧数据编码为base64字符串
        
        Args:
            frame_data: 帧图像数据
            
        Returns:
            str: base64编码的图像字符串
        """
        try:
            return base64.b64encode(frame_data).decode('utf-8')
        except Exception as e:
            logger.error(f"帧编码失败: {str(e)}")
            raise
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        获取视频基本信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            Dict: 包含视频信息的字典
        """
        try:
            import subprocess
            import json
            
            if not os.path.exists(video_path):
                logger.error(f"视频文件不存在: {video_path}")
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
                
            logger.info(f"尝试获取视频信息: {video_path}")
            
            # 记录当前路径
            current_dir = os.getcwd()
            logger.info(f"当前工作目录: {current_dir}")
            
            # 尝试使用绝对路径
            abs_path = os.path.abspath(video_path)
            logger.info(f"视频文件绝对路径: {abs_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(abs_path)
            logger.info(f"视频文件大小: {file_size} 字节")
            if file_size < 1000:
                logger.warning(f"视频文件过小，可能不是有效的视频: {file_size} 字节")
            
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,duration,r_frame_rate',
                '-of', 'json', abs_path
            ]
            
            logger.info(f"执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"ffprobe命令失败，退出码: {result.returncode}")
                logger.error(f"错误输出: {result.stderr}")
                
                # 尝试使用ffmpeg获取信息作为备选方案
                logger.info("尝试使用ffmpeg代替ffprobe获取视频信息")
                cmd_alt = [
                    'ffmpeg', '-i', abs_path, 
                    '-hide_banner'
                ]
                result_alt = subprocess.run(cmd_alt, capture_output=True, text=True, timeout=30)
                # 从stderr中解析视频信息（ffmpeg通常将视频信息输出到stderr）
                stderr = result_alt.stderr
                logger.info(f"ffmpeg输出: {stderr}")
                
                # 简单解析视频尺寸
                width, height = 0, 0
                duration = 0
                fps = 0
                
                # 尝试从stderr中提取信息
                import re
                # 尝试匹配视频尺寸，格式通常是类似 "Stream #0:0: Video: ... 1280x720"
                size_match = re.search(r'(\d+)x(\d+)', stderr)
                if size_match:
                    width = int(size_match.group(1))
                    height = int(size_match.group(2))
                    logger.info(f"从ffmpeg输出中提取到视频尺寸: {width}x{height}")
                
                # 尝试匹配时长，格式通常是类似 "Duration: 00:05:12.45"
                duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', stderr)
                if duration_match:
                    hours = int(duration_match.group(1))
                    minutes = int(duration_match.group(2))
                    seconds = float(duration_match.group(3))
                    duration = hours * 3600 + minutes * 60 + seconds
                    logger.info(f"从ffmpeg输出中提取到视频时长: {duration}秒")
                
                # 尝试匹配帧率，格式通常是类似 "fps, 30 tbr"
                fps_match = re.search(r'(\d+(?:\.\d+)?) fps', stderr)
                if fps_match:
                    fps = float(fps_match.group(1))
                    logger.info(f"从ffmpeg输出中提取到视频帧率: {fps}")
                
                return {
                    'width': width or 1280,  # 默认值
                    'height': height or 720,
                    'duration': duration or 10,
                    'fps': fps or 30
                }
            
            # 正常解析ffprobe的JSON输出
            logger.info(f"ffprobe输出: {result.stdout}")
            info = json.loads(result.stdout)
            
            stream = info.get('streams', [{}])[0]
            width = int(stream.get('width', 320))
            height = int(stream.get('height', 240))
            
            # 处理duration可能为None的情况
            duration_str = stream.get('duration')
            if duration_str is None:
                logger.warning("视频持续时间未知，使用默认值5秒")
                duration = 5
            else:
                try:
                    duration = float(duration_str)
                except ValueError:
                    logger.warning(f"无法解析时长: {duration_str}，使用默认值5秒")
                    duration = 5
            
            # 解析帧率
            fps_str = stream.get('r_frame_rate', '24/1')
            if not fps_str:
                logger.warning("视频帧率未知，使用默认值24")
                fps = 24
            elif '/' in fps_str:
                try:
                    num, den = map(int, fps_str.split('/'))
                    fps = num / den if den != 0 else 24
                except Exception as e:
                    logger.warning(f"解析帧率分数失败: {fps_str}, 错误: {str(e)}")
                    fps = 24
            else:
                try:
                    fps = float(fps_str)
                except ValueError:
                    logger.warning(f"无法解析帧率: {fps_str}，使用默认值24")
                    fps = 24
            
            video_info = {
                'width': width,
                'height': height,
                'duration': duration,
                'fps': fps
            }
            
            logger.info(f"成功获取视频信息: {video_info}")
            return video_info
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {str(e)}", exc_info=True)
            # 返回默认值
            default_info = {
                'width': 1280,
                'height': 720,
                'duration': 10,
                'fps': 30
            }
            logger.info(f"使用默认视频信息: {default_info}")
            return default_info 