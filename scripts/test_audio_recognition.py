#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试音频文件的语音识别
用法: python scripts/test_audio_recognition.py <音频文件路径>
"""

import os
import sys
from pathlib import Path
import wave
import logging

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from app.services.speech_recognition_service import SpeechRecognitionFactory

def test_audio_recognition(audio_path):
    """测试音频文件的语音识别"""
    try:
        audio_path = os.path.abspath(audio_path)
        if not os.path.exists(audio_path):
            print(f"错误: 音频文件不存在: {audio_path}")
            return
        
        print(f"正在分析音频文件: {audio_path}")
        
        # 如果是wav文件，获取音频信息
        is_wav = audio_path.lower().endswith(".wav")
        if is_wav:
            try:
                with wave.open(audio_path, 'rb') as wav_file:
                    n_frames = wav_file.getnframes()
                    framerate = wav_file.getframerate()
                    audio_duration = n_frames / framerate
                    channels = wav_file.getnchannels()
                    
                print(f"音频信息: 时长={audio_duration:.2f}秒, 采样率={framerate}Hz, 声道数={channels}, 总帧数={n_frames}")
            except Exception as e:
                print(f"读取WAV文件信息失败: {e}")
                is_wav = False
        
        # 读取音频文件
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            print(f"音频数据大小: {len(audio_data) / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"读取音频文件失败: {e}")
            return
        
        # 尝试所有可用的语音识别服务
        print("\n测试所有可用的语音识别服务...")
        
        # 1. 首先尝试Whisper
        print("\n=== 使用Whisper语音识别服务 ===")
        whisper_service = SpeechRecognitionFactory.get_service("whisper")
        if whisper_service.is_available():
            print("Whisper服务可用，开始识别...")
            
            options = {}
            if is_wav:
                options = {
                    'format': 'wav',
                    'rate': framerate
                }
            
            try:
                if is_wav:
                    result = whisper_service.recognize(audio_data, 'wav', framerate, options)
                else:
                    # 对于非WAV文件，尝试以不同采样率识别
                    for rate in [16000, 22050, 44100]:
                        print(f"尝试采样率: {rate}Hz")
                        result = whisper_service.recognize(audio_data, os.path.splitext(audio_path)[1][1:], rate, {})
                        if result and result != "[识别错误]":
                            break
                
                print("\n识别结果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
                if not result or result == "[识别错误]":
                    print("警告: Whisper未能识别出文本")
            except Exception as e:
                print(f"Whisper识别失败: {e}")
        else:
            print("Whisper服务不可用")
        
        # 2. 尝试百度语音识别
        print("\n=== 使用百度语音识别服务 ===")
        baidu_service = SpeechRecognitionFactory.get_service("baidu")
        if baidu_service.is_available():
            print("百度语音识别服务可用，开始识别...")
            
            options = {}
            if is_wav:
                options = {
                    'format': 'wav',
                    'rate': framerate
                }
            
            try:
                if is_wav:
                    result = baidu_service.recognize(audio_data, 'wav', framerate, options)
                else:
                    print("警告: 百度服务可能不支持非WAV格式")
                    result = baidu_service.recognize(audio_data, os.path.splitext(audio_path)[1][1:], 16000, {})
                
                print("\n识别结果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
                if not result or result == "[识别错误]":
                    print("警告: 百度服务未能识别出文本")
            except Exception as e:
                print(f"百度语音识别失败: {e}")
        else:
            print("百度语音识别服务不可用")
        
        # 3. 尝试Vosk语音识别
        print("\n=== 使用Vosk语音识别服务 ===")
        vosk_service = SpeechRecognitionFactory.get_service("opensource")
        if vosk_service.is_available():
            print("Vosk语音识别服务可用，开始识别...")
            
            options = {}
            if is_wav:
                options = {
                    'format': 'wav',
                    'rate': framerate
                }
            
            try:
                if is_wav:
                    result = vosk_service.recognize(audio_data, 'wav', framerate, options)
                else:
                    print("警告: Vosk可能不支持非WAV格式")
                    result = vosk_service.recognize(audio_data, 'wav', 16000, {})
                
                print("\n识别结果:")
                print("-" * 40)
                print(result)
                print("-" * 40)
                
                if not result or result == "[识别错误]":
                    print("警告: Vosk未能识别出文本")
            except Exception as e:
                print(f"Vosk语音识别失败: {e}")
        else:
            print("Vosk语音识别服务不可用")
            
    except Exception as e:
        print(f"测试过程中出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供音频文件路径")
        print("用法: python scripts/test_audio_recognition.py <音频文件路径>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    test_audio_recognition(audio_path) 