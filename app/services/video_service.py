import os
import uuid
import subprocess
import requests
import logging
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import shutil
from datetime import datetime

import speech_recognition as sr
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, ImageClip
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from aip import AipSpeech  # 添加百度语音识别SDK导入
import wave

from app.config import settings, TEMP_DIR, OUTPUT_DIR, UPLOAD_DIR
from app.core.database import Task, TaskStatus, VideoType
from app.schemas.video import VideoAnalysisResult, ChapterPoint
from app.services.ai_service import ai_service
from app.services.speech_recognition_service import SpeechRecognitionFactory

logger = logging.getLogger(__name__)

class VideoService:
    """视频处理服务"""
    
    def download_video(self, url: str) -> str:
        """下载视频文件并返回本地路径"""
        try:
            # 创建唯一文件名
            filename = f"{uuid.uuid4()}.mp4"
            filepath = UPLOAD_DIR / filename
            
            # 下载文件
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return str(filepath)
        except Exception as e:
            logger.error(f"下载视频失败: {str(e)}")
            raise
    
    def extract_audio(self, video_path: str) -> str:
        """从视频中提取音频"""
        try:
            # 创建临时音频文件路径
            audio_path = TEMP_DIR / f"{uuid.uuid4()}.wav"
            
            # 使用 FFmpeg 提取音频
            cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_path)]
            subprocess.run(cmd, check=True)
            
            return str(audio_path)
        except Exception as e:
            logger.error(f"提取音频失败: {str(e)}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> str:
        """转录音频为文本"""
        try:
            # 使用语音识别服务工厂获取服务实例
            speech_service = SpeechRecognitionFactory.get_service()
            
            # 以分块方式处理大文件
            transcript = ""
            
            # 获取音频信息
            with wave.open(audio_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                audio_duration = n_frames / framerate
            
            # 每60秒处理一段
            chunk_duration = 60  # 秒
            chunks = int(audio_duration / chunk_duration) + 1
            chunk_frames = int(framerate * chunk_duration)
            
            for i in range(chunks):
                # 计算当前块的帧范围
                start_frame = i * chunk_frames
                if start_frame >= n_frames:
                    break
                
                # 读取并处理音频块
                with wave.open(audio_path, 'rb') as wav_file:
                    wav_file.setpos(start_frame)
                    # 确保不超出文件结尾
                    frames_to_read = min(chunk_frames, n_frames - start_frame)
                    audio_data = wav_file.readframes(frames_to_read)
                
                # 调用语音识别服务
                try:
                    options = {
                        'format': 'wav',
                        'rate': framerate
                    }
                    
                    chunk_text = speech_service.recognize(audio_data, 'wav', framerate, options)
                    transcript += chunk_text + " "
                    
                except Exception as e:
                    logger.error(f"语音识别服务请求错误: {str(e)}")
                    transcript += "[识别错误] "
            
            return transcript.strip()
        except Exception as e:
            logger.error(f"音频转录失败: {str(e)}")
            # 如果出错，返回空字符串，稍后会通过大模型处理
            return ""
    
    def text_to_speech(self, text: str, output_path: str) -> str:
        """将文本转换为语音（简单实现，实际项目中应使用更高质量的TTS服务）"""
        try:
            # 使用gTTS进行简单的文本到语音转换
            from gtts import gTTS
            
            tts = gTTS(text=text, lang='zh-cn', slow=False)
            tts.save(output_path)
            
            return output_path
        except Exception as e:
            logger.error(f"文本转语音失败: {str(e)}")
            raise
    
    def create_progress_bar_frame(self, 
                                  width: int, 
                                  height: int, 
                                  current_time: float, 
                                  total_duration: float,
                                  chapter_points: List[ChapterPoint]) -> np.ndarray:
        """创建一帧进度条图像"""
        # 创建透明背景
        image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # 进度条设置
        bar_height = 10
        bar_y = height - 30
        bar_width = width - 40
        bar_x = 20
        
        # 绘制背景条
        draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                       fill=(100, 100, 100, 200))
        
        # 计算当前进度
        progress_width = int(bar_width * (current_time / total_duration))
        draw.rectangle([(bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height)], 
                       fill=(255, 0, 0, 200))
        
        # 绘制章节点
        for point in chapter_points:
            point_x = bar_x + int(bar_width * (point.time / total_duration))
            draw.rectangle([(point_x-2, bar_y-5), (point_x+2, bar_y + bar_height+5)], 
                           fill=(255, 255, 0, 200))
        
        # 绘制时间文本
        try:
            # 尝试加载中文字体，如果不可用则使用默认字体
            font = ImageFont.truetype("SimHei.ttf", 16)
        except IOError:
            font = ImageFont.load_default()
            
        current_time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
        total_time_str = f"{int(total_duration//60):02d}:{int(total_duration%60):02d}"
        draw.text((bar_x, bar_y - 20), current_time_str, fill=(255, 255, 255, 200), font=font)
        draw.text((bar_x + bar_width - 40, bar_y - 20), total_time_str, fill=(255, 255, 255, 200), font=font)
        
        # 转换为numpy数组
        return np.array(image)
    
    def create_video_with_narration(self, 
                                    video_path: str, 
                                    analysis_result: VideoAnalysisResult,
                                    task_id: str,
                                    progress_callback=None) -> str:
        """创建带解说和字幕的新视频"""
        try:
            # 加载视频
            video = VideoFileClip(video_path)
            
            # 生成解说音频
            narration_audio_path = str(TEMP_DIR / f"{task_id}_narration.mp3")
            self.text_to_speech(analysis_result.narration_script, narration_audio_path)
            narration_audio = AudioFileClip(narration_audio_path)
            
            # 创建字幕 (简单实现，每10秒一个字幕)
            subtitles = []
            script_lines = analysis_result.narration_script.split('\n')
            line_duration = narration_audio.duration / len(script_lines)
            
            for i, line in enumerate(script_lines):
                if line.strip():
                    start_time = i * line_duration
                    subtitle = TextClip(line, fontsize=24, color='white', bg_color='black',
                                       size=(video.w, None), method='caption')
                    subtitle = subtitle.set_start(start_time).set_duration(line_duration)
                    subtitles.append(subtitle)
            
            # 创建带有进度条的视频
            def make_frame(t):
                # 获取原始帧
                frame = video.get_frame(t)
                
                # 创建进度条
                progress_bar = self.create_progress_bar_frame(
                    video.w, 50, t, video.duration, analysis_result.chapter_points)
                
                # 合并帧和进度条
                if progress_bar.shape[1] == frame.shape[1]:  # 宽度一致才合并
                    h_offset = frame.shape[0] - progress_bar.shape[0]
                    frame[h_offset:, :, :3] = frame[h_offset:, :, :3] * (1 - progress_bar[:, :, 3:4] / 255.0) + \
                                            progress_bar[:, :, :3] * (progress_bar[:, :, 3:4] / 255.0)
                
                # 更新进度
                if progress_callback and t % 1 < 0.1:  # 每秒更新一次
                    progress = min(100.0, (t / video.duration) * 100)
                    progress_callback(progress)
                
                return frame
            
            # 创建最终视频
            processed_video = video.fl(make_frame)
            
            # 添加解说音轨和原音轨（降低原音轨音量）
            original_audio = video.audio.volumex(0.3) if video.audio else None
            if original_audio:
                final_audio = CompositeAudioClip([original_audio, narration_audio])
            else:
                final_audio = narration_audio
            
            processed_video = processed_video.set_audio(final_audio)
            
            # 合并字幕
            final_video = CompositeVideoClip([processed_video] + subtitles)
            
            # 输出最终视频
            output_path = str(OUTPUT_DIR / f"{task_id}_result.mp4")
            final_video.write_videofile(
                output_path, 
                codec="libx264", 
                audio_codec="aac", 
                temp_audiofile=str(TEMP_DIR / f"{task_id}_temp_audio.m4a"), 
                remove_temp=True
            )
            
            # 清理
            if os.path.exists(narration_audio_path):
                os.remove(narration_audio_path)
            
            return output_path
        except Exception as e:
            logger.error(f"创建视频失败: {str(e)}")
            raise

    def process_video(self, task: Task, db) -> None:
        """处理视频的主流程，包括转录、分析和创建新视频"""
        try:
            # 更新任务状态
            task.status = TaskStatus.PROCESSING
            task.message = "开始处理视频"
            task.progress = 5.0
            db.commit()
            
            # 下载视频（如果是URL）
            video_path = task.video_path
            if task.video_url:
                task.message = "正在下载视频"
                db.commit()
                video_path = self.download_video(task.video_url)
                task.video_path = video_path
                task.progress = 15.0
                db.commit()
            
            # 提取音频
            task.message = "正在提取音频"
            task.progress = 20.0
            db.commit()
            audio_path = self.extract_audio(video_path)
            
            # 转录音频
            task.message = "正在转录音频"
            task.progress = 30.0
            db.commit()
            transcript = self.transcribe_audio(audio_path)
            
            # 获取视频时长
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            
            # 分析内容并生成解说
            task.message = "正在分析内容并生成解说"
            task.progress = 50.0
            db.commit()
            analysis_result = ai_service.analyze_transcript(transcript, task.video_type, duration)
            
            # 创建最终视频
            task.message = "正在创建最终视频"
            task.progress = 60.0
            db.commit()
            
            # 进度回调函数
            def update_progress(progress):
                # 将进度映射到60-95区间
                mapped_progress = 60.0 + (progress * 0.35)
                task.progress = min(95.0, mapped_progress)
                task.message = f"正在创建视频...{int(progress)}%"
                db.commit()
            
            result_path = self.create_video_with_narration(
                video_path, 
                analysis_result, 
                task.id,
                progress_callback=update_progress
            )
            
            # 更新任务完成状态
            task.status = TaskStatus.COMPLETED
            task.progress = 100.0
            task.message = "视频处理完成"
            task.result_path = result_path
            db.commit()
            
            # 清理临时文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
        except Exception as e:
            logger.error(f"视频处理失败: {str(e)}")
            task.status = TaskStatus.FAILED
            task.message = f"处理失败: {str(e)}"
            db.commit()
            raise

# 创建视频服务单例
video_service = VideoService() 