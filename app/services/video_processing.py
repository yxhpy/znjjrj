#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
直接使用ffmpeg的视频处理服务
替代使用MoviePy的方式，避免递归深度超限问题
"""

import os
import uuid
import subprocess
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

from app.config import settings, TEMP_DIR, OUTPUT_DIR
from app.services.tts_service import get_tts_service
from app.schemas.video import VideoAnalysisResult, ChapterPoint

logger = logging.getLogger(__name__)

class FFmpegVideoProcessor:
    """使用FFmpeg的视频处理器"""
    
    def __init__(self):
        """初始化视频处理器"""
        self._check_ffmpeg()
    
    def _check_ffmpeg(self):
        """检查ffmpeg是否可用"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE, 
                                  text=True)
            if result.returncode == 0:
                first_line = result.stdout.split('\n')[0]
                logger.info(f"检测到ffmpeg: {first_line}")
            else:
                logger.error(f"ffmpeg命令返回错误: {result.stderr}")
                raise RuntimeError("FFmpeg命令返回错误")
        except Exception as e:
            logger.error(f"检查ffmpeg失败: {str(e)}")
            raise RuntimeError(f"FFmpeg未安装或不可用: {str(e)}")
    
    def _create_subtitle_file(self, script: str, output_path: str, duration: float) -> str:
        """创建SRT字幕文件，根据标点符号断句"""
        try:
            # 按标点符号断句，而不是按行分割
            text = script.strip().replace('\n', ' ')
            
            # 使用标点符号断句
            import re
            # 匹配中文或英文的标点符号作为断句点
            segments = re.split(r'([。！？.!?；;，,])', text)
            
            # 重组断句（保留标点符号）
            sentences = []
            i = 0
            while i < len(segments) - 1:
                if i + 1 < len(segments):
                    sentences.append(segments[i] + segments[i + 1])
                    i += 2
                else:
                    sentences.append(segments[i])
                    i += 1
            
            # 处理最后一个可能的片段
            if i < len(segments):
                sentences.append(segments[i])
            
            # 过滤掉空字符串
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                logger.warning("解说脚本为空，将创建空字幕文件")
                sentences = ["[无解说文本]"]
            
            logger.info(f"将解说文本分割为{len(sentences)}个句子")
            
            # 获取音频文件并分析语音时长
            narration_audio_path = output_path.replace(".srt", ".mp3")
            if os.path.exists(narration_audio_path):
                # 使用音频时长而不是视频时长来计算字幕
                audio_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                                '-of', 'default=noprint_wrappers=1:nokey=1', narration_audio_path]
                try:
                    audio_duration = float(subprocess.check_output(audio_info_cmd, text=True).strip())
                    logger.info(f"使用解说音频时长计算字幕: {audio_duration}秒")
                    duration = audio_duration
                except Exception as e:
                    logger.warning(f"获取音频时长失败: {str(e)}，使用提供的时长: {duration}秒")
            
            # 计算每个句子的大致时长（根据字符数量比例分配）
            total_chars = sum(len(s) for s in sentences)
            sentence_durations = []
            
            for sentence in sentences:
                # 按字符数量比例分配时间
                char_ratio = len(sentence) / total_chars
                sentence_durations.append(duration * char_ratio)
            
            # 写入SRT格式
            with open(output_path, 'w', encoding='utf-8') as f:
                current_time = 0.0
                for i, (sentence, sentence_duration) in enumerate(zip(sentences, sentence_durations)):
                    if not sentence.strip():
                        continue
                    
                    # 计算时间戳
                    start_time = current_time
                    end_time = start_time + sentence_duration
                    current_time = end_time
                    
                    # 转换为SRT格式时间戳 (HH:MM:SS,mmm)
                    start_h = int(start_time // 3600)
                    start_m = int((start_time % 3600) // 60)
                    start_s = int(start_time % 60)
                    start_ms = int((start_time % 1) * 1000)
                    
                    end_h = int(end_time // 3600)
                    end_m = int((end_time % 3600) // 60)
                    end_s = int(end_time % 60)
                    end_ms = int((end_time % 1) * 1000)
                    
                    # 字幕序号从1开始
                    f.write(f"{i+1}\n")
                    f.write(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> ")
                    f.write(f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}\n")
                    f.write(f"{sentence}\n\n")
            
            logger.info(f"创建字幕文件成功: {output_path}, 字幕条数: {len(sentences)}")
            return output_path
        except Exception as e:
            logger.error(f"创建字幕文件失败: {str(e)}")
            raise
    
    def _create_chapter_markers_file(self, chapter_points: List[ChapterPoint], output_path: str, video_duration: float) -> Optional[str]:
        """创建章节标记元数据文件 (FFmpeg元数据格式)"""
        try:
            if not chapter_points:
                logger.warning("没有章节点，跳过章节标记创建")
                return None
            
            # 确保章节点按时间排序
            sorted_points = sorted(chapter_points, key=lambda p: p.time)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(";FFMETADATA1\n")
                
                for i, point in enumerate(sorted_points):
                    # 章节开始时间 (毫秒)
                    start_time_ms = int(point.time * 1000)
                    
                    # 写入章节信息
                    f.write(f"[CHAPTER]\n")
                    f.write(f"TIMEBASE=1/1000\n")
                    f.write(f"START={start_time_ms}\n")
                    
                    # 下一章节开始前的时间或者视频结束前1毫秒
                    if i < len(sorted_points) - 1:
                        end_time_ms = int(sorted_points[i+1].time * 1000) - 1
                    else:
                        # 最后一个章节持续到视频实际结束时间
                        end_time_ms = int(video_duration * 1000) - 1
                    
                    # 确保结束时间大于开始时间
                    if end_time_ms <= start_time_ms:
                        end_time_ms = start_time_ms + 1000  # 如果结束时间不合理，设置为开始时间后1秒
                    
                    f.write(f"END={end_time_ms}\n")
                    f.write(f"title={point.title}\n\n")
            
            logger.info(f"创建章节标记文件成功: {output_path}, 章节数: {len(sorted_points)}")
            return output_path
        except Exception as e:
            logger.error(f"创建章节标记文件失败: {str(e)}")
            return None
    
    def create_video_with_narration(self, 
                                   video_path: str, 
                                   analysis_result: VideoAnalysisResult,
                                   task_id: str,
                                   progress_callback: Optional[Callable[[float], None]] = None) -> str:
        """创建带解说和字幕的新视频，并根据分析结果进行剪辑"""
        try:
            logger.info(f"开始处理视频: {video_path}")
            
            # 创建输出文件路径
            output_path = str(OUTPUT_DIR / f"{task_id}_result.mp4")
            
            # 步骤1: 生成解说音频
            if progress_callback:
                progress_callback(5.0)  # 5%进度
                
            narration_audio_path = str(TEMP_DIR / f"{task_id}_narration.mp3")
            tts_service = get_tts_service()
            tts_service.synthesize(analysis_result.narration_script, narration_audio_path)
            logger.info(f"生成解说音频成功: {narration_audio_path}")
            
            if progress_callback:
                progress_callback(15.0)  # 15%进度
            
            # 步骤2: 获取视频时长和解说音频时长
            video_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                             '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            video_duration = float(subprocess.check_output(video_info_cmd, text=True).strip())
            
            audio_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                             '-of', 'default=noprint_wrappers=1:nokey=1', narration_audio_path]
            audio_duration = float(subprocess.check_output(audio_info_cmd, text=True).strip())
            
            logger.info(f"原始视频时长: {video_duration}秒, 解说音频时长: {audio_duration}秒")
            
            if progress_callback:
                progress_callback(20.0)  # 20%进度
                
            # 步骤3: 根据分析结果进行视频剪辑
            # 检查是否有保留片段的建议
            keep_segments = []
            edited_video_path = None
            
            # 从chapter_points提取关键时刻
            key_moments = []
            if analysis_result.chapter_points:
                # 确保按时间排序
                sorted_points = sorted(analysis_result.chapter_points, key=lambda p: p.time)
                key_moments = [(point.time, point.title) for point in sorted_points]
            
            # 如果有关键时刻，根据关键时刻生成剪辑片段
            if key_moments:
                logger.info(f"根据{len(key_moments)}个关键时刻生成剪辑片段")
                
                # 为每个关键时刻创建一个片段，每个片段从该时刻开始，持续一段时间
                segment_duration = min(20.0, video_duration / len(key_moments))  # 每个片段最长20秒
                
                for i, (time_point, _) in enumerate(key_moments):
                    # 确保时间点在视频范围内
                    if time_point < video_duration:
                        # 片段开始时间
                        start_time = max(0, time_point - 2.0)  # 从关键时刻前2秒开始
                        # 片段结束时间
                        end_time = min(video_duration, start_time + segment_duration)
                        # 确保结束时间大于开始时间
                        if end_time > start_time:
                            keep_segments.append((start_time, end_time))
                
                logger.info(f"生成了{len(keep_segments)}个保留片段")
            
            # 如果没有关键时刻或片段，则根据视频时长创建若干均匀分布的片段
            if not keep_segments:
                logger.info("没有找到剪辑建议，将创建均匀分布的片段")
                # 根据解说时长确定需要的片段总时长
                segments_count = max(3, min(10, int(video_duration / 30)))  # 最少3个片段，最多10个片段
                segment_duration = min(15.0, video_duration / segments_count)  # 每个片段最长15秒
                
                for i in range(segments_count):
                    start_time = i * (video_duration / segments_count)
                    end_time = min(video_duration, start_time + segment_duration)
                    keep_segments.append((start_time, end_time))
                
                logger.info(f"创建了{segments_count}个均匀分布的片段")
            
            # 剪辑视频片段
            if keep_segments:
                # 为每个片段创建一个临时文件
                segment_files = []
                for i, (start_time, end_time) in enumerate(keep_segments):
                    segment_file = str(TEMP_DIR / f"{task_id}_segment_{i}.mp4")
                    # 使用ffmpeg剪切片段，使用更高效的方式
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(start_time),  # 放在输入前以使用关键帧查找
                        '-i', video_path,
                        '-to', str(end_time - start_time),  # 持续时间
                        '-c:v', 'libx264', '-preset', 'ultrafast',  # 使用更快的预设
                        '-c:a', 'aac',
                        '-avoid_negative_ts', '1',
                        segment_file
                    ]
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        segment_files.append(segment_file)
                        logger.info(f"成功剪切片段 {i+1}/{len(keep_segments)}: {start_time:.2f}s - {end_time:.2f}s")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"剪切片段 {i+1}/{len(keep_segments)} 失败: {e}")
                
                # 如果成功剪切了片段，创建片段列表文件
                if segment_files:
                    segments_list_file = str(TEMP_DIR / f"{task_id}_segments_list.txt")
                    with open(segments_list_file, 'w', encoding='utf-8') as f:
                        for segment_file in segment_files:
                            f.write(f"file '{segment_file}'\n")
                    
                    # 合并所有片段
                    edited_video_path = str(TEMP_DIR / f"{task_id}_edited.mp4")
                    concat_cmd = [
                        'ffmpeg', '-y',
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', segments_list_file,
                        '-c', 'copy',  # 直接复制，避免重新编码
                        edited_video_path
                    ]
                    try:
                        subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        logger.info(f"成功合并{len(segment_files)}个片段为剪辑视频")
                        
                        # 获取剪辑后视频的时长
                        edited_video_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                                               '-of', 'default=noprint_wrappers=1:nokey=1', edited_video_path]
                        edited_video_duration = float(subprocess.check_output(edited_video_info_cmd, text=True).strip())
                        logger.info(f"剪辑后视频时长: {edited_video_duration}秒")
                        
                        # 更新视频路径为剪辑后的视频
                        video_path = edited_video_path
                        video_duration = edited_video_duration
                    except subprocess.CalledProcessError as e:
                        logger.error(f"合并视频片段失败: {e}")
                        # 如果合并失败，继续使用原始视频
                        logger.info("将使用原始视频继续处理")
            
            if progress_callback:
                progress_callback(40.0)  # 40%进度
            
            # 确定最终视频时长
            final_duration = max(video_duration, audio_duration)
            # 更新: 确保最终视频时长比音频略长一些，防止音频播放未完视频就结束
            if audio_duration > video_duration:
                final_duration = audio_duration * 1.05  # 添加5%的缓冲时间
            logger.info(f"最终视频时长: {final_duration}秒")
            
            # 步骤4: 创建字幕文件
            subtitle_path = str(TEMP_DIR / f"{task_id}_subtitles.srt")
            self._create_subtitle_file(analysis_result.narration_script, subtitle_path, final_duration)
            
            if progress_callback:
                progress_callback(50.0)  # 50%进度
            
            # 步骤5: 创建章节标记文件 (如果有章节点)
            chapter_file = None
            if analysis_result.chapter_points:
                chapter_file = str(TEMP_DIR / f"{task_id}_chapters.txt")
                self._create_chapter_markers_file(analysis_result.chapter_points, chapter_file, final_duration)
            
            if progress_callback:
                progress_callback(60.0)  # 60%进度
            
            # 步骤6: 处理视频
            # 如果需要延长视频 (解说音频比视频长)
            if final_duration > video_duration:
                # 创建一个临时文件，通过loop选项延长视频时长
                extended_video_path = str(TEMP_DIR / f"{task_id}_extended.mp4")
                extend_cmd = [
                    'ffmpeg', '-y',
                    '-stream_loop', '-1',  # 循环视频
                    '-i', video_path,
                    '-t', str(final_duration),  # 设置时长
                    '-c:v', 'libx264', '-preset', 'medium',  # 使用较好的编码质量
                    '-vsync', 'cfr',  # 保持恒定帧率
                    '-r', '30',       # 设置固定帧率
                    extended_video_path
                ]
                try:
                    subprocess.run(extend_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    logger.info(f"视频延长成功: {video_duration}秒 -> {final_duration}秒")
                    # 更新视频路径指向延长后的视频
                    video_path = extended_video_path
                    video_duration = final_duration
                except subprocess.CalledProcessError as e:
                    logger.error(f"视频延长失败: {e}")
                    logger.info("将使用原始视频继续处理")
            # 如果视频比音频长，调整视频速度
            elif video_duration > audio_duration * 1.1:  # 如果视频比音频长超过10%
                # 创建一个临时文件，调整视频速度
                speed_adjusted_path = str(TEMP_DIR / f"{task_id}_speed_adjusted.mp4")
                # 计算速度比例 (视频时长/音频时长)
                speed_factor = video_duration / audio_duration
                logger.info(f"视频时长是音频时长的{speed_factor:.2f}倍，将调整视频速度")
                
                # 使用setpts过滤器调整视频速度
                speed_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-filter_complex', f"[0:v]setpts={1/speed_factor}*PTS[v]",
                    '-map', '[v]',
                    '-c:v', 'libx264', '-preset', 'medium',
                    '-an',  # 不包含音频
                    speed_adjusted_path
                ]
                try:
                    subprocess.run(speed_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    # 检查调整后的视频时长
                    adjusted_video_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                                             '-of', 'default=noprint_wrappers=1:nokey=1', speed_adjusted_path]
                    adjusted_duration = float(subprocess.check_output(adjusted_video_info_cmd, text=True).strip())
                    logger.info(f"视频速度调整成功: {video_duration}秒 -> {adjusted_duration}秒")
                    
                    # 更新视频路径指向速度调整后的视频
                    video_path = speed_adjusted_path
                    video_duration = adjusted_duration
                except subprocess.CalledProcessError as e:
                    logger.error(f"调整视频速度失败: {e}")
                    logger.info("将使用原始视频继续处理")
            # 如果视频略长于音频，裁剪视频
            elif video_duration > audio_duration:
                # 创建一个临时文件，裁剪视频时长
                trimmed_video_path = str(TEMP_DIR / f"{task_id}_trimmed.mp4")
                trim_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-t', str(audio_duration),  # 设置时长与音频一致
                    '-c:v', 'libx264', '-preset', 'medium',  # 使用较好的编码质量
                    '-vsync', 'cfr',  # 保持恒定帧率
                    '-an',           # 没有音频
                    trimmed_video_path
                ]
                try:
                    subprocess.run(trim_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    logger.info(f"视频裁剪成功: {video_duration}秒 -> {audio_duration}秒")
                    # 更新视频路径指向裁剪后的视频
                    video_path = trimmed_video_path
                    video_duration = audio_duration
                except subprocess.CalledProcessError as e:
                    logger.error(f"视频裁剪失败: {e}")
                    logger.info("将使用原始视频继续处理")
            
            if progress_callback:
                progress_callback(70.0)  # 70%进度
            
            # 处理步骤分解为多个阶段，避免一次性使用过多内存
            # 步骤7.1: 先混合音频
            mixed_audio_path = str(TEMP_DIR / f"{task_id}_mixed_audio.aac")
            has_original_audio = False
            
            try:
                # 检查原视频是否有音频
                audio_check_cmd = [
                    'ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 
                    'stream=codec_type', '-of', 'csv=p=0', video_path
                ]
                audio_check = subprocess.run(audio_check_cmd, stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE, text=True)
                
                if 'audio' in audio_check.stdout:
                    has_original_audio = True
                    logger.info("原视频包含音频，将与解说混合")
                    
                    # 混合音频文件
                    audio_mix_cmd = [
                        'ffmpeg', '-y',
                        '-i', narration_audio_path,  # 解说音频
                        '-i', video_path,            # 原视频（用于提取音频）
                        '-filter_complex', '[0:a]volume=1.0[a1];[1:a]volume=0.0[a2];[a1][a2]amix=inputs=2:duration=longest',
                        '-c:a', 'aac', '-b:a', '192k',
                        mixed_audio_path
                    ]
                    subprocess.run(audio_mix_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    logger.info("音频混合成功")
                else:
                    # 直接使用解说音频
                    logger.info("原视频不包含音频，直接使用解说音频")
                    audio_convert_cmd = [
                        'ffmpeg', '-y',
                        '-i', narration_audio_path,
                        '-c:a', 'aac', '-b:a', '192k',
                        mixed_audio_path
                    ]
                    subprocess.run(audio_convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            except Exception as e:
                logger.warning(f"音频处理失败: {str(e)}, 将只使用解说音频")
                # 直接使用解说音频
                audio_convert_cmd = [
                    'ffmpeg', '-y',
                    '-i', narration_audio_path,
                    '-c:a', 'aac', '-b:a', '192k',
                    mixed_audio_path
                ]
                subprocess.run(audio_convert_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            # 步骤7.2: 添加字幕
            subtitled_video_path = str(TEMP_DIR / f"{task_id}_subtitled.mp4")
            subtitle_cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vf', f"subtitles='{subtitle_path}':force_style='FontName=SimHei,FontSize=18,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=3,Outline=1,Shadow=0,Alignment=2,MarginV=30'",
                '-c:v', 'libx264', '-preset', 'medium',
                '-vsync', 'cfr',  # 保持恒定帧率
                '-r', '30',       # 固定帧率
                '-an',  # 不包含音频
                subtitled_video_path
            ]
            subprocess.run(subtitle_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            logger.info("添加字幕成功")
            
            if progress_callback:
                progress_callback(85.0)  # 85%进度
            
            # 步骤7.3: 合并带字幕的视频和混合音频
            # 准备ffmpeg命令
            final_cmd = [
                'ffmpeg', '-y',
                '-i', subtitled_video_path,  # 带字幕的视频
                '-i', mixed_audio_path,      # 混合后的音频
            ]
            
            # 如果有章节标记，使用单独的命令处理
            if chapter_file:
                # 添加章节元数据
                final_cmd.extend([
                    '-i', chapter_file,
                    '-map', '0:v',
                    '-map', '1:a',
                    '-map_metadata', '2',
                ])
            else:
                final_cmd.extend([
                    '-map', '0:v',
                    '-map', '1:a',
                ])
            
            # 获取混合音频的准确时长
            audio_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                            '-of', 'default=noprint_wrappers=1:nokey=1', mixed_audio_path]
            try:
                audio_final_duration = float(subprocess.check_output(audio_info_cmd, text=True).strip())
                logger.info(f"最终音频时长: {audio_final_duration}秒")
                
                # 设置输出参数
                final_cmd.extend([
                    '-c:v', 'libx264',     # 重新编码视频以确保同步
                    '-preset', 'medium',   # 使用较好的编码质量
                    '-c:a', 'aac',         # 重新编码音频以确保同步
                    '-b:a', '192k',        # 设置音频比特率
                    '-vsync', 'cfr',       # 保持恒定帧率
                    '-r', '30',            # 设置固定帧率
                    '-shortest',           # 以最短输入流决定输出时长
                    '-avoid_negative_ts', '1',  # 避免负时间戳
                    '-async', '1',         # 音频同步
                    # 添加精确的时长参数，设为音频时长
                    '-t', str(audio_final_duration),  # 确保最终视频时长与音频时长一致
                    # 设置元数据
                    '-metadata', f'title=AI解说视频_{task_id}',
                    '-metadata', 'comment=由AI自动生成的解说视频',
                    output_path
                ])
            except Exception as e:
                logger.warning(f"获取最终音频时长失败: {str(e)}，使用默认命令")
                # 设置基本输出参数
                final_cmd.extend([
                    '-c:v', 'libx264',  # 重新编码视频
                    '-preset', 'medium',
                    '-c:a', 'aac',      # 重新编码音频
                    '-b:a', '192k',     # 设置音频比特率
                    '-vsync', 'cfr',    # 保持恒定帧率
                    '-r', '30',         # 设置固定帧率
                    '-shortest',        # 以最短输入流决定输出时长
                    '-async', '1',      # 音频同步
                    output_path
                ])
            
            # 执行命令
            logger.info(f"执行最终FFmpeg命令: {' '.join(final_cmd)}")
            subprocess.run(final_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            
            if progress_callback:
                progress_callback(95.0)  # 95%进度
            
            logger.info(f"视频处理成功，输出文件: {output_path}")
            
            # 清理临时文件
            temp_files = [
                narration_audio_path, 
                subtitle_path, 
                mixed_audio_path,
                subtitled_video_path
            ]
            
            if chapter_file:
                temp_files.append(chapter_file)
            if 'extended_video_path' in locals() and os.path.exists(locals().get('extended_video_path', '')):
                temp_files.append(extended_video_path)
            if 'speed_adjusted_path' in locals() and os.path.exists(locals().get('speed_adjusted_path', '')):
                temp_files.append(locals().get('speed_adjusted_path'))
            if 'trimmed_video_path' in locals() and 'trimmed_video_path' in locals() and os.path.exists(locals().get('trimmed_video_path', '')):
                temp_files.append(locals().get('trimmed_video_path'))
            if 'edited_video_path' in locals() and edited_video_path and os.path.exists(edited_video_path):
                temp_files.append(edited_video_path)
            if 'segments_list_file' in locals() and 'segments_list_file' in locals():
                temp_files.append(locals().get('segments_list_file'))
            
            # 清理片段文件
            if 'segment_files' in locals() and segment_files:
                temp_files.extend(segment_files)
                
            for file in temp_files:
                if file and os.path.exists(file):
                    try:
                        os.remove(file)
                        logger.info(f"删除临时文件: {file}")
                    except Exception as e:
                        logger.warning(f"删除临时文件失败: {file}, 错误: {str(e)}")
            
            if progress_callback:
                progress_callback(100.0)  # 100%进度
                
            return output_path
        except Exception as e:
            logger.error(f"使用ffmpeg处理视频失败: {str(e)}")
            raise

# 创建视频处理器单例
ffmpeg_video_processor = FFmpegVideoProcessor() 