#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
长视频处理脚本

用于处理长电影或教学视频，提供命令行接口，支持配置分段大小和并行处理
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import time
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings, TEMP_DIR, OUTPUT_DIR
from app.core.database import VideoType
from app.services.video_service import VideoService
from app.services.ai_service import AIService

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='长视频处理工具')
    
    parser.add_argument(
        '--video_path', '-v', 
        type=str, 
        required=True,
        help='视频文件路径'
    )
    
    parser.add_argument(
        '--video_type', '-t', 
        type=str, 
        choices=['movie', 'course'], 
        default='movie',
        help='视频类型: movie(电影) 或 course(教学视频)'
    )
    
    parser.add_argument(
        '--chunk_size', '-c', 
        type=int, 
        default=settings.TRANSCRIPT_CHUNK_SIZE,
        help='转录文本分片大小(字符数)'
    )
    
    parser.add_argument(
        '--output', '-o', 
        type=str, 
        default=None,
        help='输出视频路径，默认为原文件名加上_processed后缀'
    )
    
    parser.add_argument(
        '--save_transcript', '-s', 
        action='store_true',
        help='是否保存转录文本'
    )
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 验证视频文件存在
    video_path = os.path.abspath(args.video_path)
    if not os.path.exists(video_path):
        logger.error(f"视频文件不存在: {video_path}")
        sys.exit(1)
    
    # 设置输出路径
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_processed.mp4")
    
    # 创建临时任务ID
    task_id = f"cli_{int(time.time())}"
    
    # 视频类型
    video_type = VideoType.MOVIE if args.video_type == 'movie' else VideoType.COURSE
    
    # 设置分片大小
    if args.chunk_size != settings.TRANSCRIPT_CHUNK_SIZE:
        logger.info(f"使用自定义分片大小: {args.chunk_size}")
        settings.TRANSCRIPT_CHUNK_SIZE = args.chunk_size
    
    # 创建服务实例
    video_service = VideoService()
    
    try:
        logger.info("开始处理长视频...")
        logger.info(f"视频路径: {video_path}")
        logger.info(f"视频类型: {args.video_type}")
        
        # 提取音频
        logger.info("提取音频...")
        audio_path = video_service.extract_audio(video_path)
        
        # 转录音频
        logger.info("转录音频（这可能需要一些时间）...")
        transcript = video_service.transcribe_audio(audio_path)
        
        # 保存转录文本（如果需要）
        if args.save_transcript:
            transcript_path = os.path.join(TEMP_DIR, f"{task_id}_transcript.txt")
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            logger.info(f"已保存转录文本: {transcript_path}")
        
        # 获取视频时长
        logger.info("获取视频信息...")
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()
        
        # 分析内容并生成解说
        logger.info("分析内容并生成解说（对于长视频，这一步可能需要较长时间）...")
        ai_service = AIService()
        analysis_result = ai_service.analyze_transcript(transcript, video_type, duration)
        
        # 保存分析结果
        analysis_path = os.path.join(TEMP_DIR, f"{task_id}_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_result.dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"已保存分析结果: {analysis_path}")
        
        # 创建最终视频
        logger.info("创建最终视频...")
        
        # 进度回调函数
        def update_progress(progress):
            logger.info(f"视频处理进度: {int(progress)}%")
        
        result_path = video_service.create_video_with_narration(
            video_path, 
            analysis_result, 
            task_id,
            progress_callback=update_progress
        )
        
        # 如果结果不在指定位置，移动到目标位置
        if result_path != output_path:
            import shutil
            shutil.move(result_path, output_path)
            result_path = output_path
        
        logger.info(f"视频处理完成！输出路径: {result_path}")
        
        # 清理临时文件
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
    except Exception as e:
        logger.error(f"处理失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 