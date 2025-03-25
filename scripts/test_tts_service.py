#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试TTS服务的脚本
"""

import os
import sys
import logging
import time

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.tts_service import get_tts_service

logging.basicConfig(level=logging.INFO)

def main():
    start_time = time.time()
    print("开始测试TTS服务...")
    
    # 获取TTS服务
    tts_service = get_tts_service()
    
    # 测试文本
    text = "这是一个测试文本，用于验证TTS服务是否正常工作。这段文字将被转换为语音。"
    
    # 创建输出目录
    os.makedirs("temp", exist_ok=True)
    output_path = os.path.join("temp", "tts_test_output.wav")
    
    # 合成语音
    tts_service.synthesize(text, output_path)
    
    print(f"语音合成完成，输出文件：{output_path}")
    print(f"耗时：{time.time() - start_time:.2f}秒")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
