"""
使用FFmpeg的关键帧提取器
"""
import os
import tempfile
import subprocess
from typing import List
import logging
import imghdr
import shutil
from pathlib import Path
from datetime import datetime

from .base import FrameExtractor

logger = logging.getLogger(__name__)

class FFmpegFrameExtractor(FrameExtractor):
    """使用FFmpeg提取关键帧的实现"""
    
    def extract_frames(self, video_path: str) -> List[str]:
        """
        使用FFmpeg从视频中提取关键帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            List[str]: base64编码的帧图像列表
            
        Raises:
            Exception: 当视频文件不存在或处理失败时抛出
        """
        # 确保视频文件存在且可读
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise Exception(f"视频文件不存在: {video_path}")
        if not os.path.isfile(video_path):
            raise Exception(f"视频路径不是文件: {video_path}")
        if not os.access(video_path, os.R_OK):
            raise Exception(f"视频文件无法读取: {video_path}")
            
        try:
            # 提前验证视频文件格式
            try:
                video_info = self.get_video_info(video_path)
                duration = video_info['duration']
                logger.info(f"视频信息: 分辨率={video_info['width']}x{video_info['height']}, "
                          f"时长={duration}秒, 帧率={video_info['fps']}")
            except Exception as e:
                logger.error(f"获取视频信息失败: {str(e)}")
                raise Exception(f"无法读取视频信息，可能是无效或损坏的文件: {str(e)}")
            
            # 创建输出目录用于调试
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("output", f"frames_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"帧提取结果将保存到: {output_dir}")
            
            # 保存视频信息到输出目录
            with open(os.path.join(output_dir, "video_info.txt"), 'w') as f:
                f.write(f"视频路径: {video_path}\n")
                f.write(f"视频信息: {video_info}\n")
            
            # 创建临时目录存储帧
            with tempfile.TemporaryDirectory() as temp_dir:
                frames_base64 = []
                
                # 计算时间间隔，确保最小时间间隔为1秒
                max_frames = min(self.max_frames, int(duration))
                if max_frames <= 0:
                    max_frames = 1
                
                interval = duration / (max_frames + 1)
                
                # 提取每个时间点的帧
                for i in range(1, max_frames + 1):
                    time_point = interval * i
                    output_file = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
                    debug_file = os.path.join(output_dir, f"frame_{i:03d}.jpg")
                    
                    logger.info(f"尝试提取时间点 {time_point:.2f}s 的帧")
                    
                    # 使用ffmpeg提取帧
                    cmd = [
                        'ffmpeg', '-y',  # 放在前面确保覆盖输出文件
                        '-ss', str(time_point),  # 设置时间点
                        '-i', video_path,  # 输入文件
                        '-frames:v', '1',  # 只提取1帧
                        '-q:v', '2',  # 设置质量级别为2(1-31,1为最高)
                        output_file  # 输出文件
                    ]
                    
                    # 记录命令
                    cmd_str = ' '.join(cmd)
                    logger.info(f"执行命令: {cmd_str}")
                    with open(os.path.join(output_dir, f"command_{i}.txt"), 'w') as f:
                        f.write(cmd_str)
                    
                    # 执行命令
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        with open(os.path.join(output_dir, f"ffmpeg_output_{i}.txt"), 'w') as f:
                            f.write(f"退出码: {result.returncode}\n\n标准输出:\n{result.stdout}\n\n错误输出:\n{result.stderr}")
                        
                        if result.returncode != 0:
                            logger.warning(f"FFmpeg命令返回非零退出码: {result.returncode}")
                            logger.warning(f"错误输出: {result.stderr}")
                            continue
                    except subprocess.TimeoutExpired:
                        logger.error("FFmpeg命令执行超时")
                        continue
                    except Exception as e:
                        logger.error(f"FFmpeg命令执行失败: {str(e)}")
                        continue
                    
                    # 检查文件是否创建成功
                    if not os.path.exists(output_file):
                        logger.error(f"输出文件不存在: {output_file}")
                        continue
                        
                    if os.path.getsize(output_file) == 0:
                        logger.error(f"输出文件为空: {output_file}")
                        continue
                    
                    # 复制文件到输出目录进行检查
                    try:
                        shutil.copy(output_file, debug_file)
                        logger.info(f"保存帧到: {debug_file}")
                    except Exception as e:
                        logger.error(f"复制文件失败: {str(e)}")
                    
                    # 检查图片格式
                    try:
                        img_format = imghdr.what(output_file)
                        if img_format is None:
                            logger.warning(f"无法识别图片格式，但继续处理: {output_file}")
                        else:
                            logger.info(f"检测到图片格式: {img_format}")
                    except Exception as e:
                        logger.error(f"检查图片格式失败: {str(e)}")
                    
                    # 读取图片并编码为base64
                    try:
                        with open(output_file, 'rb') as f:
                            img_data = f.read()
                        img_str = self.encode_frame(img_data)
                        frames_base64.append(img_str)
                        logger.info(f"成功提取时间点 {time_point:.2f}s 的帧")
                    except Exception as e:
                        logger.error(f"读取或编码图片失败: {str(e)}")
                        continue
                
                # 若没有提取到任何帧，尝试最简单的方法提取一帧
                if not frames_base64:
                    logger.warning("尝试最简单的方法提取一帧")
                    output_file = os.path.join(temp_dir, "simple_frame.jpg")
                    debug_file = os.path.join(output_dir, "simple_frame.jpg")
                    
                    # 使用最简单的参数
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-frames:v', '1',
                        output_file
                    ]
                    
                    # 记录命令
                    cmd_str = ' '.join(cmd)
                    logger.info(f"执行命令: {cmd_str}")
                    with open(os.path.join(output_dir, "simple_command.txt"), 'w') as f:
                        f.write(cmd_str)
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        with open(os.path.join(output_dir, "simple_ffmpeg_output.txt"), 'w') as f:
                            f.write(f"退出码: {result.returncode}\n\n标准输出:\n{result.stdout}\n\n错误输出:\n{result.stderr}")
                        
                        if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            # 复制文件到输出目录
                            shutil.copy(output_file, debug_file)
                            
                            # 读取图片并编码为base64
                            with open(output_file, 'rb') as f:
                                img_data = f.read()
                            img_str = self.encode_frame(img_data)
                            frames_base64.append(img_str)
                            logger.info(f"使用简单方法成功提取一帧")
                    except Exception as e:
                        logger.error(f"简单方法提取帧失败: {str(e)}")
                
                # 如果还是没有提取到帧，则尝试使用-ss选项放在输入文件后面
                if not frames_base64:
                    logger.warning("尝试使用-ss在输入文件后的方法提取帧")
                    output_file = os.path.join(temp_dir, "alt_frame.jpg")
                    debug_file = os.path.join(output_dir, "alt_frame.jpg")
                    
                    # 在输入文件后使用-ss
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', video_path,
                        '-ss', '5',  # 跳到第5秒
                        '-frames:v', '1',
                        output_file
                    ]
                    
                    # 记录命令
                    cmd_str = ' '.join(cmd)
                    logger.info(f"执行命令: {cmd_str}")
                    with open(os.path.join(output_dir, "alt_command.txt"), 'w') as f:
                        f.write(cmd_str)
                    
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                        with open(os.path.join(output_dir, "alt_ffmpeg_output.txt"), 'w') as f:
                            f.write(f"退出码: {result.returncode}\n\n标准输出:\n{result.stdout}\n\n错误输出:\n{result.stderr}")
                        
                        if result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            # 复制文件到输出目录
                            shutil.copy(output_file, debug_file)
                            
                            # 读取图片并编码为base64
                            with open(output_file, 'rb') as f:
                                img_data = f.read()
                            img_str = self.encode_frame(img_data)
                            frames_base64.append(img_str)
                            logger.info(f"使用替代方法成功提取一帧")
                    except Exception as e:
                        logger.error(f"替代方法提取帧失败: {str(e)}")
                
                if not frames_base64:
                    error_msg = "未能提取任何帧"
                    logger.error(error_msg)
                    
                    # 在抛出异常前，生成一个详细的错误报告
                    with open(os.path.join(output_dir, "error_report.txt"), 'w') as f:
                        f.write(f"视频路径: {video_path}\n")
                        f.write(f"视频信息: {video_info}\n")
                        f.write(f"错误: {error_msg}\n")
                    
                    raise Exception(error_msg)
                
                logger.info(f"成功提取 {len(frames_base64)} 帧，保存在 {output_dir}")
                return frames_base64
                
        except Exception as e:
            logger.error(f"FFmpeg提取帧失败: {str(e)}")
            raise 