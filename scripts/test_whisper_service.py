#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试Whisper语音识别服务
"""

import os
import sys
import argparse
import tempfile
import subprocess
import wave
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.speech_recognition_service import SpeechRecognitionFactory

def generate_test_audio():
    """
    生成测试音频文件
    """
    print("生成测试音频文件...")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # 音频参数
    duration = 3  # 秒
    sample_rate = 16000  # 采样率
    
    # 生成一个简单的正弦波
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    
    # 写入WAV文件
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    print(f"测试音频文件已生成: {temp_path}")
    return temp_path

def record_test_audio(duration=5):
    """
    使用系统麦克风录制测试音频
    """
    try:
        # 检查是否有ffmpeg
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告: 未安装ffmpeg，无法录制音频。将生成测试音频文件。")
        return generate_test_audio()
    
    print(f"录制测试音频 ({duration}秒)...")
    print("请对着麦克风说话...")
    
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    # 使用ffmpeg录制音频
    try:
        subprocess.run([
            "ffmpeg", "-y", 
            "-f", "dshow" if sys.platform == "win32" else "alsa", 
            "-i", "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{00000000-0000-0000-0000-000000000000}" if sys.platform == "win32" else "default",
            "-t", str(duration),
            "-ar", "16000",
            "-ac", "1",
            temp_path
        ], check=True)
        
        print(f"录音完成: {temp_path}")
        return temp_path
        
    except subprocess.CalledProcessError as e:
        print(f"录音失败: {e}")
        print("将生成测试音频文件。")
        return generate_test_audio()

def test_whisper_service(audio_path=None, duration=5, options=None):
    """
    测试Whisper语音识别服务
    """
    # 获取音频文件
    if audio_path is None:
        audio_path = record_test_audio(duration)
    else:
        print(f"使用提供的音频文件: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"错误: 音频文件不存在: {audio_path}")
        return
    
    # 读取音频文件
    try:
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
    except Exception as e:
        print(f"读取音频文件失败: {e}")
        return
    
    # 创建识别服务
    print("初始化Whisper语音识别服务...")
    service = SpeechRecognitionFactory.get_service("whisper")
    
    if not service.is_available():
        print("错误: Whisper语音识别服务不可用")
        print("请先运行: python scripts/download_whisper_model.py")
        return
    
    # 设置选项
    if options is None:
        options = {}
    
    # 开始识别
    print("正在识别音频...")
    result = service.recognize(audio_data, "wav", 16000, options)
    
    print("\n识别结果:")
    print("-" * 40)
    print(result)
    print("-" * 40)
    
    # 清理临时文件
    if audio_path.startswith(tempfile.gettempdir()):
        try:
            os.unlink(audio_path)
            print(f"临时音频文件已删除: {audio_path}")
        except Exception as e:
            print(f"无法删除临时文件: {e}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="测试Whisper语音识别服务")
    parser.add_argument("--audio", help="输入音频文件路径 (不提供则录制)")
    parser.add_argument("--duration", type=int, default=5, help="录制音频时长(秒)")
    parser.add_argument("--language", default="zh", help="识别语言")
    
    args = parser.parse_args()
    
    # 构建选项
    options = {
        "language": args.language,
        "beam_size": 5
    }
    
    # 测试识别服务
    test_whisper_service(args.audio, args.duration, options)

if __name__ == "__main__":
    main() 