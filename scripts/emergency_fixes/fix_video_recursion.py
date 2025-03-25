#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频处理递归问题应急修复脚本

使用直接的ffmpeg命令替代MoviePy处理视频，
避免递归深度超限问题
"""

import os
import sys
import logging
import tempfile
import subprocess
import uuid
from pathlib import Path
import shutil
import json

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import settings, TEMP_DIR, OUTPUT_DIR
from app.services.tts_service import get_tts_service
from app.schemas.video import VideoAnalysisResult, ChapterPoint

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    """检查ffmpeg是否可用"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE, 
                               text=True)
        if result.returncode == 0:
            # 修复f-string中的反斜杠问题
            first_line = result.stdout.split('\n')[0]
            logger.info(f"检测到ffmpeg: {first_line}")
            return True
        else:
            logger.error(f"ffmpeg命令返回错误: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"检查ffmpeg失败: {str(e)}")
        return False

def create_subtitle_file(script, output_path, duration):
    """创建SRT字幕文件"""
    try:
        lines = script.strip().split('\n')
        if not lines:
            logger.warning("解说脚本为空，将创建空字幕文件")
            lines = ["[无解说文本]"]
        
        # 根据视频时长计算每行字幕的持续时间
        line_duration = duration / len(lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                # 计算字幕时间戳
                start_time = i * line_duration
                end_time = min((i + 1) * line_duration, duration)
                
                # 转换为SRT格式时间戳 (HH:MM:SS,mmm)
                start_h = int(start_time // 3600)
                start_m = int((start_time % 3600) // 60)
                start_s = int(start_time % 60)
                start_ms = int((start_time % 1) * 1000)
                
                end_h = int(end_time // 3600)
                end_m = int((end_time % 3600) // 60)
                end_s = int(end_time % 60)
                end_ms = int((end_time % 1) * 1000)
                
                # 写入SRT格式字幕
                f.write(f"{i+1}\n")
                f.write(f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d} --> ")
                f.write(f"{end_h:02d}:{end_m:02d}:{end_s:02d},{end_ms:03d}\n")
                f.write(f"{line}\n\n")
        
        logger.info(f"创建字幕文件成功: {output_path}, 字幕条数: {len(lines)}")
        return output_path
    except Exception as e:
        logger.error(f"创建字幕文件失败: {str(e)}")
        return None

def create_chapter_markers_file(chapter_points, output_path):
    """创建章节标记元数据文件 (FFmpeg元数据格式)"""
    try:
        if not chapter_points:
            logger.warning("没有章节点，跳过章节标记创建")
            return None
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(";FFMETADATA1\n")
            
            for i, point in enumerate(chapter_points):
                # 章节开始时间 (毫秒)
                start_time_ms = int(point.time * 1000)
                
                # 写入章节信息
                f.write(f"[CHAPTER]\n")
                f.write(f"TIMEBASE=1/1000\n")
                f.write(f"START={start_time_ms}\n")
                # 下一章节开始前的时间或者视频结束前1毫秒
                if i < len(chapter_points) - 1:
                    end_time_ms = int(chapter_points[i+1].time * 1000) - 1
                else:
                    # 最后一个章节持续到视频结束，但这里我们设置一个足够长的时间（1小时）
                    end_time_ms = start_time_ms + 3600000
                
                f.write(f"END={end_time_ms}\n")
                f.write(f"title={point.title}\n\n")
        
        logger.info(f"创建章节标记文件成功: {output_path}, 章节数: {len(chapter_points)}")
        return output_path
    except Exception as e:
        logger.error(f"创建章节标记文件失败: {str(e)}")
        return None

def process_video_with_ffmpeg(video_path, analysis_result, task_id):
    """使用ffmpeg处理视频"""
    try:
        logger.info(f"开始使用ffmpeg处理视频: {video_path}")
        
        # 检查ffmpeg是否可用
        if not check_ffmpeg():
            raise RuntimeError("未检测到ffmpeg，请先安装ffmpeg")
        
        # 创建输出文件路径
        output_path = str(OUTPUT_DIR / f"{task_id}_result.mp4")
        
        # 步骤1: 生成解说音频
        narration_audio_path = str(TEMP_DIR / f"{task_id}_narration.mp3")
        tts_service = get_tts_service()
        tts_service.synthesize(analysis_result.narration_script, narration_audio_path)
        logger.info(f"生成解说音频成功: {narration_audio_path}")
        
        # 步骤2: 获取视频时长和解说音频时长
        video_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                         '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        video_duration = float(subprocess.check_output(video_info_cmd, text=True).strip())
        
        audio_info_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                         '-of', 'default=noprint_wrappers=1:nokey=1', narration_audio_path]
        audio_duration = float(subprocess.check_output(audio_info_cmd, text=True).strip())
        
        logger.info(f"视频时长: {video_duration}秒, 解说音频时长: {audio_duration}秒")
        
        # 确定最终视频时长
        final_duration = max(video_duration, audio_duration)
        logger.info(f"最终视频时长: {final_duration}秒")
        
        # 步骤3: 创建字幕文件
        subtitle_path = str(TEMP_DIR / f"{task_id}_subtitles.srt")
        create_subtitle_file(analysis_result.narration_script, subtitle_path, final_duration)
        
        # 步骤4: 创建章节标记文件 (如果有章节点)
        chapter_file = None
        if analysis_result.chapter_points:
            chapter_file = str(TEMP_DIR / f"{task_id}_chapters.txt")
            create_chapter_markers_file(analysis_result.chapter_points, chapter_file)
        
        # 步骤5: 处理视频
        # 如果需要延长视频 (解说音频比视频长)
        if final_duration > video_duration:
            # 创建一个临时文件，通过loop选项延长视频时长
            extended_video_path = str(TEMP_DIR / f"{task_id}_extended.mp4")
            extend_cmd = [
                'ffmpeg', '-y',
                '-stream_loop', '-1',  # 循环视频
                '-i', video_path,
                '-t', str(final_duration),  # 设置时长
                '-c:v', 'libx264', '-preset', 'medium',
                extended_video_path
            ]
            subprocess.run(extend_cmd, check=True)
            logger.info(f"视频延长成功: {video_duration}秒 -> {final_duration}秒")
            # 更新视频路径指向延长后的视频
            video_path = extended_video_path
        
        # 步骤6: 合并视频、解说音频和字幕
        # 准备ffmpeg命令
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,            # 输入视频
            '-i', narration_audio_path,  # 解说音频
        ]
        
        # 如果有章节标记，添加章节文件输入
        if chapter_file:
            cmd.extend(['-i', chapter_file])
        
        # 尝试从原视频提取音频 (如果有)
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
                
                # 提取原始音频
                original_audio_path = str(TEMP_DIR / f"{task_id}_original_audio.aac")
                extract_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-vn', '-acodec', 'copy',
                    original_audio_path
                ]
                subprocess.run(extract_cmd, check=True)
                
                # 添加原始音频作为另一个输入
                cmd.extend(['-i', original_audio_path])
        except Exception as e:
            logger.warning(f"检查原视频音频失败: {str(e)}, 将只使用解说音频")
        
        # 添加字幕
        cmd.extend([
            '-vf', f"subtitles='{subtitle_path}'",  # 添加字幕
        ])
        
        # 设置音频混合
        if has_original_audio:
            # 混合原始音频(音量降低70%)和解说音频
            last_audio_index = 3 if chapter_file else 2
            cmd.extend([
                '-filter_complex', f'[1:a]volume=1.0[a1];[{last_audio_index}:a]volume=0.3[a2];[a1][a2]amix=inputs=2:duration=longest',
            ])
        else:
            # 只使用解说音频
            cmd.extend(['-map', '0:v', '-map', '1:a'])
        
        # 设置输出参数
        cmd.extend([
            '-c:v', 'libx264', '-preset', 'medium',
            '-c:a', 'aac', '-b:a', '192k',
            '-shortest',  # 以最短输入流决定输出时长
        ])
        
        # 如果有章节标记，添加元数据映射
        if chapter_file:
            chapter_index = 2
            cmd.extend(['-map_metadata', str(chapter_index)])
        
        # 添加输出文件
        cmd.append(output_path)
        
        # 执行命令
        logger.info(f"执行FFmpeg命令: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        logger.info(f"视频处理成功，输出文件: {output_path}")
        
        # 清理临时文件
        temp_files = [narration_audio_path, subtitle_path]
        if chapter_file:
            temp_files.append(chapter_file)
        if 'extended_video_path' in locals():
            temp_files.append(extended_video_path)
        if 'original_audio_path' in locals() and has_original_audio:
            temp_files.append(original_audio_path)
            
        for file in temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    logger.info(f"删除临时文件: {file}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {file}, 错误: {str(e)}")
        
        return output_path
    except Exception as e:
        logger.error(f"使用ffmpeg处理视频失败: {str(e)}")
        raise

def main():
    """主函数"""
    try:
        # 解析命令行参数
        if len(sys.argv) < 2:
            print("用法: python fix_video_recursion.py <视频文件路径> [分析结果JSON文件路径]")
            return
        
        video_path = sys.argv[1]
        
        # 检查视频文件是否存在
        if not os.path.exists(video_path):
            print(f"错误: 视频文件不存在: {video_path}")
            return
        
        # 任务ID
        task_id = f"emergency_fix_{uuid.uuid4()}"
        
        # 分析结果
        if len(sys.argv) >= 3:
            # 从文件加载分析结果
            analysis_json_path = sys.argv[2]
            with open(analysis_json_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            # 构建VideoAnalysisResult对象
            chapter_points = [ChapterPoint(**cp) for cp in analysis_data.get('chapter_points', [])]
            analysis_result = VideoAnalysisResult(
                narration_script=analysis_data.get('narration_script', '这是一个应急处理生成的视频。'),
                chapter_points=chapter_points,
                video_type=analysis_data.get('video_type', 'MOVIE'),
                duration=analysis_data.get('duration', 0),
                transcript=analysis_data.get('transcript', '')
            )
        else:
            # 创建一个简单的默认分析结果
            analysis_result = VideoAnalysisResult(
                narration_script="这是一个应急处理生成的视频。\n原始视频可能存在格式问题，已使用ffmpeg应急处理。",
                chapter_points=[],
                video_type="MOVIE",
                duration=0,
                transcript=""
            )
        
        # 处理视频
        output_path = process_video_with_ffmpeg(video_path, analysis_result, task_id)
        print(f"视频处理成功，输出文件: {output_path}")
        
    except Exception as e:
        print(f"应急处理失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 