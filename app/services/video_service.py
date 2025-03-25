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
from moviepy.config import change_settings

from app.config import settings, TEMP_DIR, OUTPUT_DIR, UPLOAD_DIR
from app.core.database import Task, TaskStatus, VideoType
from app.schemas.video import VideoAnalysisResult, ChapterPoint
from app.services.ai_service import ai_service
from app.services.speech_recognition_service import SpeechRecognitionFactory
from app.services.tts_service import get_tts_service
from app.services.video_processing import ffmpeg_video_processor  # 导入新的处理器
from app.services.omni_service import omni_service

logger = logging.getLogger(__name__)

class VideoService:
    """视频处理服务"""
    
    def __init__(self):
        """初始化视频服务"""
        # 确保MoviePy能找到ImageMagick
        import os
        from moviepy.config import change_settings
        
        # 尝试查找ImageMagick的convert命令位置
        try:
            import subprocess
            result = subprocess.run(['which', 'convert'], stdout=subprocess.PIPE, text=True)
            if result.returncode == 0 and result.stdout.strip():
                convert_path = result.stdout.strip()
                logger.info(f"找到ImageMagick convert路径: {convert_path}")
                change_settings({"IMAGEMAGICK_BINARY": convert_path})
            else:
                # 尝试常见位置
                for path in ['/usr/bin/convert', '/usr/local/bin/convert']:
                    if os.path.exists(path):
                        logger.info(f"使用默认ImageMagick convert路径: {path}")
                        change_settings({"IMAGEMAGICK_BINARY": path})
                        break
        except Exception as e:
            logger.warning(f"设置ImageMagick路径时出错: {str(e)}")
    
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
            # 确保使用绝对路径
            audio_path = os.path.abspath(audio_path)
            if not os.path.isfile(audio_path):
                logger.error(f"音频文件不存在: {audio_path}")
                return "[音频文件不存在]"
            
            logger.info(f"开始转录音频文件: {audio_path}")
            
            # 检查音频时长
            try:
                # 使用ffprobe检查音频时长
                cmd_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
                duration_result = subprocess.run(cmd_duration, capture_output=True, text=True)
                
                try:
                    duration = float(duration_result.stdout.strip())
                    logger.info(f"音频文件总时长: {duration:.2f}秒")
                    
                    if duration < 0.5:
                        logger.warning(f"警告: 音频文件过短 ({duration:.2f}秒)，可能影响识别质量")
                except ValueError:
                    logger.warning(f"无法解析音频时长: {duration_result.stdout}")
            except Exception as e:
                logger.warning(f"检查音频时长失败: {str(e)}")
                
            # 使用语音识别服务工厂获取服务实例
            speech_service = SpeechRecognitionFactory.get_service()
            
            # 以分块方式处理大文件
            transcript = ""
            
            # 获取音频信息
            with wave.open(audio_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                framerate = wav_file.getframerate()
                audio_duration = n_frames / framerate
                
            logger.info(f"音频文件信息: 时长={audio_duration:.2f}秒, 采样率={framerate}Hz, 总帧数={n_frames}")
            
            # 每60秒处理一段
            chunk_duration = 60  # 秒
            chunks = int(audio_duration / chunk_duration) + 1
            chunk_frames = int(framerate * chunk_duration)
            
            logger.info(f"音频将被分为{chunks}块进行处理，每块约{chunk_duration}秒")
            
            # 使用ffmpeg分割音频而不是直接使用wave模块读取
            for i in range(chunks):
                # 计算当前块的时间范围
                start_time = i * chunk_duration
                if start_time >= audio_duration:
                    break
                
                end_time = min((i + 1) * chunk_duration, audio_duration)
                duration = end_time - start_time
                
                # 创建临时文件存储音频块
                chunk_path = f"{audio_path}_chunk_{i}.wav"
                try:
                    # 使用ffmpeg提取音频片段
                    cmd = [
                        "ffmpeg", "-y", 
                        "-i", audio_path, 
                        "-ss", str(start_time), 
                        "-t", str(duration), 
                        "-acodec", "pcm_s16le", 
                        "-ar", str(framerate), 
                        "-ac", "1", 
                        chunk_path
                    ]
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    
                    # 验证生成的音频块
                    if os.path.exists(chunk_path) and os.path.getsize(chunk_path) > 44:  # WAV头至少44字节
                        logger.info(f"成功提取第{i+1}/{chunks}块音频，起始时间={start_time:.2f}秒，时长={duration:.2f}秒")
                        
                        # 读取音频数据
                        with open(chunk_path, "rb") as f:
                            audio_data = f.read()
                            
                        # 检查是否是静音
                        is_silence = False
                        try:
                            with wave.open(chunk_path, 'rb') as wav_file:
                                frames = wav_file.readframes(wav_file.getnframes())
                                if frames and len(frames) > 0:
                                    # 将PCM数据转换为numpy数组
                                    import numpy as np
                                    pcm_data = np.frombuffer(frames, dtype=np.int16)
                                    # 检查音量级别
                                    rms = np.sqrt(np.mean(pcm_data.astype(np.float32)**2))
                                    logger.info(f"音频块{i+1}的均方根值(音量): {rms:.2f}")
                                    if rms < 10:  # 非常低的音量阈值
                                        is_silence = True
                                        logger.warning(f"音频块{i+1}可能是静音 (RMS={rms:.2f})")
                        except Exception as e:
                            logger.warning(f"无法分析音频块{i+1}的音量: {str(e)}")
                        
                        # 如果检测到是静音，可以跳过识别
                        if is_silence:
                            logger.info(f"跳过第{i+1}块音频识别 (静音)")
                            continue
                        
                        # 调用语音识别服务
                        try:
                            options = {
                                'format': 'wav',
                                'rate': framerate
                            }
                            
                            chunk_text = speech_service.recognize(audio_data, 'wav', framerate, options)
                            logger.info(f"第{i+1}块音频识别结果长度: {len(chunk_text)} 字符")
                            
                            if chunk_text and len(chunk_text) > 0:
                                transcript += chunk_text + " "
                                logger.info(f"第{i+1}块音频识别结果: {chunk_text[:100]}...")
                            else:
                                logger.warning(f"第{i+1}块音频未识别出文本")
                            
                        except Exception as e:
                            logger.error(f"语音识别服务请求错误: {str(e)}")
                            transcript += "[识别错误] "
                    else:
                        logger.warning(f"第{i+1}块音频提取失败或文件过小")
                except Exception as e:
                    logger.error(f"提取第{i+1}块音频时出错: {str(e)}")
                finally:
                    # 清理临时文件
                    if os.path.exists(chunk_path):
                        try:
                            os.remove(chunk_path)
                        except:
                            pass
            
            # 记录最终转录结果
            result = transcript.strip()
            logger.info(f"音频转录完成，总文本长度: {len(result)} 字符")
            if result:
                logger.info(f"转录结果预览: {result[:200]}...")
            else:
                logger.warning("未能提取任何文本，转录结果为空")
                
            return result
        except Exception as e:
            logger.error(f"音频转录失败: {str(e)}")
            # 如果出错，返回空字符串，稍后会通过大模型处理
            return ""
    
    def text_to_speech(self, text: str, output_path: str) -> str:
        """将文本转换为语音"""
        try:
            # 使用TTS服务进行文本到语音转换
            tts_service = get_tts_service()
            return tts_service.synthesize(text, output_path)
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
        
        # 进度条设置 - 更加美观的设计
        bar_height = 6  # 更细的进度条
        bar_y = height - 15  # 放置在更底部的位置
        bar_width = width - 100  # 两侧留出更多空间
        bar_x = 50  # 左侧留出空间
        
        # 绘制背景半透明黑色区域
        draw.rectangle([(0, bar_y - 15), (width, height)], 
                      fill=(0, 0, 0, 150))  # 半透明黑色背景
        
        # 绘制背景条 - 使用圆角矩形
        # 由于PIL不直接支持圆角矩形，我们使用普通矩形但颜色更柔和
        draw.rectangle([(bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height)], 
                       fill=(150, 150, 150, 180), outline=(180, 180, 180, 200))
        
        # 计算当前进度
        progress_width = int(bar_width * (current_time / total_duration))
        # 进度条使用渐变色
        if progress_width > 0:
            draw.rectangle([(bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height)], 
                          fill=(65, 105, 225, 220))  # 使用皇家蓝色，更美观
        
        # 绘制章节点 - 使用更明显但不突兀的标记
        for point in chapter_points:
            point_x = bar_x + int(bar_width * (point.time / total_duration))
            # 使用小圆点而非矩形
            draw.ellipse([(point_x-3, bar_y-3), (point_x+3, bar_y-3+bar_height+6)], 
                        fill=(255, 215, 0, 230))  # 金色标记点
        
        # 绘制时间文本 - 尝试加载更好的字体
        try:
            # 尝试多种可能的中文字体路径
            font_paths = [
                '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # Droid Sans
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',             # 文泉驿微米黑
                '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',     # Noto Sans CJK
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',     # Noto Sans CJK (opentype)
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'             # DejaVu Sans
            ]
            
            font = None
            for font_path in font_paths:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, 14)  # 小一点的字体
                    break
                    
            if font is None:
                font = ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
            
        # 格式化时间文本，显示更详细
        current_time_str = f"{int(current_time//60):02d}:{int(current_time%60):02d}"
        total_time_str = f"{int(total_duration//60):02d}:{int(total_duration%60):02d}"
        
        # 在进度条下方绘制当前时间和总时间
        draw.text((bar_x, bar_y + bar_height + 2), current_time_str, 
                 fill=(220, 220, 220, 230), font=font)  # 亮白色
        
        # 右对齐总时间
        total_time_width = font.getbbox(total_time_str)[2] if hasattr(font, 'getbbox') else 40
        draw.text((bar_x + bar_width - total_time_width, bar_y + bar_height + 2), 
                 total_time_str, fill=(220, 220, 220, 230), font=font)
        
        # 转换为numpy数组
        return np.array(image)
    
    def create_video_with_narration(self, 
                                    video_path: str, 
                                    analysis_result: VideoAnalysisResult,
                                    task_id: str,
                                    progress_callback=None) -> str:
        """创建带解说和字幕的新视频"""
        try:
            logger.info(f"开始处理视频: {video_path}")
            logger.info("使用FFmpeg视频处理器，避免递归深度超限问题")
            
            # 使用ffmpeg视频处理器代替MoviePy处理
            return ffmpeg_video_processor.create_video_with_narration(
                video_path, 
                analysis_result, 
                task_id, 
                progress_callback
            )
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
            
            # 检查是否应该使用全模态分析（可以通过任务参数控制）
            use_omni = getattr(task, 'use_omni', False)
            
            # 分析内容并生成解说
            if use_omni:
                task.message = "正在分析内容并生成解说"
                task.progress = 45.0
                db.commit()
                
                try:
                    task.message = "正在使用全模态模型分析视频内容和画面"
                    db.commit()
                    analysis_result = self.generate_enhanced_analysis(video_path, task.id, db)
                    logger.info("成功使用全模态模型增强分析")
                    
                    task.message = "正在处理全模态分析结果"
                    task.progress = 50.0
                    db.commit()
                except Exception as e:
                    logger.error(f"全模态模型分析失败，回退到标准分析: {str(e)}")
                    task.message = "全模态分析失败，使用标准分析继续"
                    task.progress = 45.0
                    db.commit()
                    analysis_result = ai_service.analyze_transcript(transcript, task.video_type, duration)
            else:
                task.message = "正在分析内容并生成解说"
                task.progress = 50.0
                db.commit()
                analysis_result = ai_service.analyze_transcript(transcript, task.video_type, duration)
            
            # 创建最终视频
            task.message = "正在创建最终视频"
            task.progress = 60.0
            db.commit()
            
            # 定义进度更新回调函数
            def update_progress(progress):
                # 将进度映射到60-95区间
                task.progress = 60.0 + (progress / 100.0) * 35.0
                task.message = f"正在创建最终视频 ({progress:.1f}%)"
                db.commit()
            
            # 创建视频
            output_path = self.create_video_with_narration(
                video_path,
                analysis_result,
                task.id,
                update_progress
            )
            
            # 完成任务
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.message = "视频处理完成"
            task.updated_at = datetime.utcnow()
            task.result_path = output_path
            db.commit()
            
            logger.info(f"视频处理完成：{task.id}, 输出：{output_path}")
        except Exception as e:
            # 处理错误
            logger.error(f"处理视频时出错: {str(e)}")
            task.status = TaskStatus.FAILED
            task.message = f"处理失败: {str(e)}"
            task.progress = 0.0
            db.commit()

    def analyze_with_omni(self, video_path: str, transcript: str = None) -> Dict[str, Any]:
        """使用阿里云全模态模型分析视频内容
        
        Args:
            video_path: 视频文件路径
            transcript: 视频转录文本(可选)
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        try:
            logger.info(f"使用阿里云全模态模型分析视频: {video_path}")
            
            # 检查是否有视频转录
            if not transcript:
                logger.info("未提供转录文本，尝试提取音频并转录")
                audio_path = self.extract_audio(video_path)
                transcript = self.transcribe_audio(audio_path)
            
            # 获取视频时长
            clip = VideoFileClip(video_path)
            duration = clip.duration
            clip.close()
            
            # 首先使用全模态模型生成视频内容摘要和上下文
            context_analysis = omni_service.generate_summary_with_context(
                video_path=video_path, 
                transcript=transcript,
                duration=duration
            )
            
            # 获取视频剪辑建议
            editing_suggestions = omni_service.suggest_video_edits(
                video_path=video_path,
                transcript=transcript
            )
            
            # 合并结果
            result = {
                "video_analysis": context_analysis,
                "editing_suggestions": editing_suggestions,
                "transcript": transcript,
                "duration": duration
            }
            
            logger.info("全模态模型分析完成")
            return result
        except Exception as e:
            logger.error(f"全模态模型分析失败: {str(e)}")
            raise

    def generate_enhanced_analysis(self, video_path: str, task_id: str, db=None) -> VideoAnalysisResult:
        """生成增强的视频分析结果，结合文本转录和视觉分析
        
        Args:
            video_path: 视频文件路径
            task_id: 任务ID
            db: 数据库会话(可选)
            
        Returns:
            VideoAnalysisResult: 视频分析结果
        """
        try:
            logger.info(f"为任务{task_id}生成增强分析")
            
            # 提取音频
            logger.info(f"从视频提取音频: {video_path}")
            audio_path = self.extract_audio(video_path)
            
            # 转录音频
            logger.info(f"开始转录音频: {audio_path}")
            transcript = self.transcribe_audio(audio_path)
            logger.info(f"音频转录完成，转录文本长度: {len(transcript)} 字符")
            
            # 获取视频时长
            logger.info("获取视频时长")
            video = VideoFileClip(video_path)
            duration = video.duration
            video.close()
            logger.info(f"视频时长: {duration} 秒")
            
            # 尝试使用全模态模型分析
            try:
                logger.info("开始使用全模态模型分析视频内容...")
                omni_result = self.analyze_with_omni(video_path, transcript)
                logger.info("全模态模型分析完成")
                
                # 使用全模态分析结果创建VideoAnalysisResult
                if "video_analysis" in omni_result and isinstance(omni_result["video_analysis"], dict):
                    logger.info("从全模态分析结果中提取视频分析信息")
                    analysis = omni_result["video_analysis"]
                    
                    # 提取视频类型 (默认为MOVIE)
                    video_type = VideoType.MOVIE
                    
                    # 从分析结果中提取摘要
                    summary = ""
                    if "detailed_summary" in analysis:
                        summary = analysis["detailed_summary"]
                        logger.info(f"找到详细摘要，长度: {len(summary)} 字符")
                    elif "summary" in analysis:
                        summary = analysis["summary"]
                        logger.info(f"找到摘要，长度: {len(summary)} 字符")
                    else:
                        logger.warning("分析结果中未找到摘要信息")
                    
                    # 从分析结果中提取关键点
                    key_points = []
                    if "themes" in analysis and isinstance(analysis["themes"], list):
                        key_points.extend(analysis["themes"])
                        logger.info(f"找到主题列表: {len(analysis['themes'])} 项")
                    elif "themes" in analysis and isinstance(analysis["themes"], str):
                        key_points.append(analysis["themes"])
                        logger.info("找到主题字符串")
                    
                    if "key_points" in analysis and isinstance(analysis["key_points"], list):
                        key_points.extend(analysis["key_points"])
                        logger.info(f"找到关键点列表: {len(analysis['key_points'])} 项")
                    
                    logger.info(f"总共提取了 {len(key_points)} 个关键点")
                    
                    # 提取章节点
                    chapter_points = []
                    if "editing_suggestions" in omni_result and "key_moments" in omni_result["editing_suggestions"]:
                        moments = omni_result["editing_suggestions"]["key_moments"]
                        logger.info(f"找到 {len(moments) if isinstance(moments, list) else 0} 个关键时刻")
                        
                        if isinstance(moments, list):
                            for moment in moments:
                                if isinstance(moment, dict) and "time" in moment and "description" in moment:
                                    try:
                                        time_point = float(moment["time"])
                                        chapter_points.append(
                                            ChapterPoint(
                                                time=time_point, 
                                                title=moment["description"]
                                            )
                                        )
                                    except (ValueError, TypeError) as e:
                                        logger.error(f"处理关键时刻时出错: {str(e)}, 数据: {moment}")
                    
                    logger.info(f"总共提取了 {len(chapter_points)} 个章节点")
                    
                    # 生成解说脚本
                    logger.info("开始从全模态分析结果生成解说脚本")
                    narration_script = self._generate_narration_from_omni(omni_result)
                    logger.info(f"解说脚本生成完成，长度: {len(narration_script)} 字符")
                    
                    # 创建结果对象
                    result = VideoAnalysisResult(
                        summary=summary,
                        key_points=key_points,
                        chapter_points=chapter_points,
                        narration_script=narration_script,
                        video_type=video_type,
                        duration=duration,
                        transcript=transcript
                    )
                    
                    logger.info("全模态增强分析完成")
                    return result
                else:
                    logger.warning("全模态分析结果中未找到video_analysis字段或格式不正确")
                    raise ValueError("全模态分析结果格式不符合预期")
            except Exception as omni_error:
                logger.error(f"全模态分析失败，错误详情: {str(omni_error)}")
                if db:
                    try:
                        task = db.query(Task).filter(Task.id == task_id).first()
                        if task:
                            task.message = f"全模态分析失败: {str(omni_error)[:100]}，回退到标准分析"
                            db.commit()
                    except Exception as db_error:
                        logger.error(f"更新任务状态失败: {str(db_error)}")
            
            # 如果全模态分析失败，回退到标准分析
            logger.info("使用标准AI分析视频内容")
            return ai_service.analyze_transcript(transcript, VideoType.MOVIE, duration)
        except Exception as e:
            logger.error(f"生成增强分析失败: {str(e)}")
            raise

    def _generate_narration_from_omni(self, omni_result: Dict[str, Any]) -> str:
        """从全模态分析结果生成解说脚本
        
        Args:
            omni_result: 全模态分析结果
            
        Returns:
            str: 解说脚本
        """
        try:
            # 使用AI服务生成解说词
            prompt = f"""基于下面的视频分析，创建一个引人入胜的解说词脚本:
            
视频分析: {json.dumps(omni_result["video_analysis"], ensure_ascii=False)}

剪辑建议: {json.dumps(omni_result["editing_suggestions"], ensure_ascii=False)}

请创建一个流畅、有趣且信息丰富的解说词，以配合视频内容。解说词应当:
1. 包含引言和结论
2. 突出视频的关键点和主题
3. 按照视频的时间顺序描述内容
4. 使用生动、吸引人的语言
5. 配合视频中的关键场景
            """
            
            # 调用AI服务
            messages = [
                {"role": "system", "content": "你是一位视频解说专家，擅长根据视频分析创建吸引人的解说词。"},
                {"role": "user", "content": prompt}
            ]
            
            response = ai_service._call_model(messages)
            narration_script = response["choices"][0]["message"]["content"]
            
            return narration_script
        except Exception as e:
            logger.error(f"从全模态结果生成解说词失败: {str(e)}")
            # 返回一个基本的解说词
            if "detailed_summary" in omni_result.get("video_analysis", {}):
                return omni_result["video_analysis"]["detailed_summary"]
            elif "summary" in omni_result.get("video_analysis", {}):
                return omni_result["video_analysis"]["summary"]
            else:
                return "无法生成解说词。"

# 创建视频服务单例
video_service = VideoService() 