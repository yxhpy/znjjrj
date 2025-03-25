#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试应急修复脚本
"""

import os
import sys
import logging
import json
import uuid
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.config import TEMP_DIR, OUTPUT_DIR
from app.core.database import VideoType
from app.schemas.video import VideoAnalysisResult, ChapterPoint
from scripts.test_video_service import create_test_video

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        # 创建测试视频
        logger.info("创建测试视频...")
        test_video_path = create_test_video()
        
        # 创建测试的分析结果
        logger.info("创建测试分析结果...")
        chapter_points = [
            ChapterPoint(time=1.0, title="章节1"),
            ChapterPoint(time=3.0, title="章节2")
        ]
        
        analysis_result = VideoAnalysisResult(
            narration_script="这是第一行测试文本。\n这是第二行测试文本。\n这是第三行测试文本。",
            chapter_points=chapter_points,
            video_type=VideoType.MOVIE,  # 假设是电影类型
            duration=5.0,                # 测试视频时长5秒
            transcript="这是原视频的文字记录"
        )
        
        # 保存分析结果到临时文件
        analysis_path = os.path.join(TEMP_DIR, f"test_analysis_{uuid.uuid4()}.json")
        
        # 转换为可序列化的字典
        analysis_dict = {
            "narration_script": analysis_result.narration_script,
            "chapter_points": [{"time": cp.time, "title": cp.title} for cp in analysis_result.chapter_points],
            "video_type": analysis_result.video_type,
            "duration": analysis_result.duration,
            "transcript": analysis_result.transcript
        }
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"分析结果已保存到: {analysis_path}")
        
        # 运行应急修复脚本
        logger.info("执行应急修复脚本...")
        from fix_video_recursion import process_video_with_ffmpeg
        
        task_id = f"test_{uuid.uuid4()}"
        try:
            output_path = process_video_with_ffmpeg(test_video_path, analysis_result, task_id)
            logger.info(f"应急修复测试成功，输出文件: {output_path}")
            
            # 验证输出文件
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                logger.info(f"输出视频文件大小: {file_size} 字节")
                
                if file_size > 1000:  # 简单检查文件大小是否合理
                    logger.info("测试通过: 输出文件大小合理")
                else:
                    logger.warning("测试警告: 输出文件大小可能过小")
            else:
                logger.error("测试失败: 输出文件不存在")
        
        except Exception as e:
            logger.error(f"应急修复测试失败: {str(e)}")
        
        # 清理临时文件
        logger.info("清理临时文件...")
        for file_path in [test_video_path, analysis_path]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"已删除临时文件: {file_path}")
                except Exception as e:
                    logger.warning(f"删除临时文件失败: {file_path}, 错误: {e}")
        
    except Exception as e:
        logger.error(f"测试应急修复失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 