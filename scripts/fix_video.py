#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
视频处理修复工具

使用FFmpeg直接处理视频，避免递归深度超限问题
用法: python fix_video.py <视频文件路径> [输出路径]
"""

import os
import sys
import logging
import json
import argparse
import uuid
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, TEMP_DIR, OUTPUT_DIR
from app.services.tts_service import get_tts_service
from app.schemas.video import VideoAnalysisResult, ChapterPoint
from app.services.video_processing import ffmpeg_video_processor
from app.services.ai_service import ai_service
from app.core.database import VideoType

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='视频处理修复工具 - 使用FFmpeg处理视频，避免递归深度超限问题')
    parser.add_argument('video_path', help='要处理的视频文件路径')
    parser.add_argument('-o', '--output', help='输出视频文件路径 (默认为output目录下的fixed_视频文件名)')
    parser.add_argument('-t', '--text', help='解说文本文件路径 (如果不提供，将使用AI生成)')
    parser.add_argument('-s', '--skip-analysis', action='store_true', help='跳过AI分析，直接使用提供的文本')
    parser.add_argument('-v', '--video-type', choices=['MOVIE', 'TUTORIAL', 'GAMEPLAY', 'DOCUMENTARY'], 
                      default='DOCUMENTARY', help='视频类型 (默认为DOCUMENTARY)')
    return parser.parse_args()

def get_video_duration(video_path):
    """获取视频时长"""
    import subprocess
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
          '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    result = subprocess.check_output(cmd, text=True).strip()
    return float(result)

def read_text_file(file_path):
    """读取文本文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_audio(video_path):
    """从视频中提取音频"""
    audio_path = str(TEMP_DIR / f"{uuid.uuid4()}.wav")
    import subprocess
    cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
          '-ar', '16000', '-ac', '1', audio_path]
    subprocess.run(cmd, check=True)
    return audio_path

def main():
    """主函数"""
    args = parse_args()
    
    # 检查视频文件是否存在
    if not os.path.exists(args.video_path):
        logger.error(f"错误: 视频文件不存在: {args.video_path}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output:
        output_path = args.output
    else:
        video_name = os.path.basename(args.video_path)
        output_path = str(OUTPUT_DIR / f"fixed_{video_name}")
    
    # 生成任务ID
    task_id = f"fix_{uuid.uuid4()}"
    
    # 创建分析结果
    if args.text and os.path.exists(args.text):
        # 使用提供的文本文件
        narration_script = read_text_file(args.text)
        logger.info(f"使用提供的文本文件: {args.text}, 长度: {len(narration_script)} 字符")
        
        # 创建简单的分析结果
        video_duration = get_video_duration(args.video_path)
        
        # 如果需要AI分析
        if not args.skip_analysis:
            logger.info("正在使用AI分析文本...")
            # 提取音频用于转录
            audio_path = extract_audio(args.video_path)
            
            # 使用AI服务进行分析
            from app.services.speech_recognition_service import SpeechRecognitionFactory
            speech_service = SpeechRecognitionFactory.get_service()
            
            # 读取音频数据
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # 转录
            transcript = speech_service.recognize(audio_data, 'wav', 16000)
            
            # 删除临时音频文件
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            # 使用AI分析内容
            analysis_result = ai_service.analyze_transcript(
                transcript,
                VideoType(args.video_type),
                video_duration
            )
            
            # 使用提供的解说文本替换AI生成的解说文本
            analysis_result.narration_script = narration_script
        else:
            # 创建简单的分析结果
            analysis_result = VideoAnalysisResult(
                narration_script=narration_script,
                chapter_points=[],  # 空章节点
                summary="通过修复工具处理的视频",
                key_points=["直接使用提供的解说文本"],
                video_type=VideoType(args.video_type),
                duration=video_duration,
                transcript=""
            )
    else:
        # 没有提供文本文件，使用AI分析
        logger.info("未提供文本文件，将使用AI生成解说文本...")
        
        # 提取音频用于转录
        audio_path = extract_audio(args.video_path)
        
        # 使用AI服务进行分析
        from app.services.speech_recognition_service import SpeechRecognitionFactory
        speech_service = SpeechRecognitionFactory.get_service()
        
        # 读取音频数据
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 转录
        transcript = speech_service.recognize(audio_data, 'wav', 16000)
        logger.info(f"转录结果长度: {len(transcript)} 字符")
        
        # 删除临时音频文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        # 获取视频时长
        video_duration = get_video_duration(args.video_path)
        
        # 使用AI分析内容
        analysis_result = ai_service.analyze_transcript(
            transcript,
            VideoType(args.video_type),
            video_duration
        )
    
    # 使用FFmpeg处理器处理视频
    try:
        logger.info("开始处理视频...")
        
        # 创建简单的进度显示回调
        def show_progress(progress):
            print(f"\r处理进度: {progress:.1f}%", end="", flush=True)
        
        # 处理视频
        final_output_path = ffmpeg_video_processor.create_video_with_narration(
            args.video_path,
            analysis_result,
            task_id,
            show_progress
        )
        
        # 如果有指定的输出路径，复制到那里
        if final_output_path != output_path:
            import shutil
            shutil.copy2(final_output_path, output_path)
            logger.info(f"已复制视频到指定输出路径: {output_path}")
        
        print()  # 换行
        logger.info(f"视频处理成功，输出文件: {output_path}")
        
    except Exception as e:
        logger.error(f"处理视频失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 