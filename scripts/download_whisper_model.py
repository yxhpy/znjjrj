#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载Whisper语音识别模型
支持两种实现：
1. faster-whisper - 基于CTranslate2的加速版本
2. whisper.cpp - C++实现的更轻量级版本
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODELS_DIR, settings

# Whisper模型目录
WHISPER_DIR = MODELS_DIR / "whisper"
WHISPER_CPP_DIR = MODELS_DIR / "whisper.cpp"

# 确保目录存在
WHISPER_DIR.mkdir(exist_ok=True, parents=True)
WHISPER_CPP_DIR.mkdir(exist_ok=True, parents=True)

def install_packages(packages):
    """
    安装Python包
    """
    print(f"正在安装所需的Python包: {', '.join(packages)}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
        return True
    except subprocess.CalledProcessError as e:
        print(f"安装包失败: {e}")
        return False

def download_faster_whisper_model(model_size="base"):
    """
    下载并准备faster-whisper模型
    """
    print(f"正在使用faster-whisper下载'{model_size}'模型...")
    
    # 安装必要的包
    if not install_packages(["faster-whisper", "soundfile"]):
        print("无法安装必要的Python包")
        return False
    
    try:
        # 使用faster-whisper下载模型
        from faster_whisper import WhisperModel
        
        print(f"下载并加载faster-whisper '{model_size}'模型 (首次运行可能需要几分钟)...")
        # 这一步会自动下载模型到huggingface缓存目录
        model = WhisperModel(model_size, device="cpu", compute_type="int8", local_files_only=False)
        
        print(f"faster-whisper '{model_size}'模型已成功下载和初始化")
        
        # 获取并显示模型路径
        import huggingface_hub
        model_path = huggingface_hub.snapshot_download(f"guillaumekln/faster-whisper-{model_size}")
        print(f"模型保存位置: {model_path}")
        
        return True
    
    except Exception as e:
        print(f"下载faster-whisper模型失败: {str(e)}")
        return False

def download_whisper_cpp_model(model_size="base"):
    """
    下载whisper.cpp模型
    """
    print(f"正在下载whisper.cpp '{model_size}'模型...")
    
    # 模型URL映射
    model_urls = {
        "tiny": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "base": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "small": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "medium": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "large": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin",
    }
    
    if model_size not in model_urls:
        print(f"错误: 未知的模型大小 '{model_size}'")
        return False
    
    url = model_urls[model_size]
    output_path = WHISPER_CPP_DIR / f"ggml-{model_size}.bin"
    
    if output_path.exists():
        print(f"模型已存在: {output_path}")
        return True
    
    try:
        # 安装requests和tqdm
        if not install_packages(["requests"]):
            print("无法安装requests包")
            return False
        
        import requests
        
        # 下载模型文件
        print(f"从 {url} 下载模型到 {output_path}...")
        
        # 开始下载
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc="下载进度",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        
        print(f"whisper.cpp '{model_size}'模型已成功下载到: {output_path}")
        return True
    
    except Exception as e:
        print(f"下载whisper.cpp模型失败: {str(e)}")
        if output_path.exists():
            output_path.unlink()  # 删除不完整的文件
        return False

def install_whisper_cpp():
    """
    安装whisper_cpp Python包
    """
    print("尝试安装whisper-cpp-python包...")
    return install_packages(["whisper-cpp-python"])

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="下载Whisper语音识别模型")
    parser.add_argument("--model-size", choices=["tiny", "base", "small", "medium", "large"], 
                        default=settings.WHISPER_MODEL_SIZE,
                        help="模型大小 (默认: base)")
    parser.add_argument("--implementation", choices=["faster-whisper", "whisper.cpp", "both"], 
                        default="both",
                        help="使用哪种实现 (默认: both)")
    
    args = parser.parse_args()
    
    success = True
    
    # 下载faster-whisper模型
    if args.implementation in ["faster-whisper", "both"]:
        if not download_faster_whisper_model(args.model_size):
            success = False
    
    # 下载whisper.cpp模型
    if args.implementation in ["whisper.cpp", "both"]:
        if not download_whisper_cpp_model(args.model_size):
            success = False
        
        if not install_whisper_cpp():
            print("警告: 无法安装whisper-cpp-python包")
    
    if success:
        print("\n所有模型下载成功!")
        # 更新.env文件中的模型路径
        update_env_file(args.model_size)
    else:
        print("\n一些模型下载失败，请检查错误信息并重试。")
        sys.exit(1)

def update_env_file(model_size):
    """
    更新.env文件中的Whisper设置
    """
    env_file = Path(__file__).resolve().parent.parent / ".env"
    
    if not env_file.exists():
        print(f"警告: .env文件不存在: {env_file}")
        return
    
    # 读取当前.env内容
    with open(env_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 需要更新的配置项
    whisper_configs = {
        "WHISPER_MODEL_SIZE": model_size,
        "WHISPER_CPP_MODEL_PATH": str(WHISPER_CPP_DIR / f"ggml-{model_size}.bin"),
        "SPEECH_RECOGNITION_SERVICE_TYPE": "whisper"
    }
    
    # 检查并更新配置
    updated_lines = []
    config_found = {key: False for key in whisper_configs}
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            updated_lines.append(line)
            continue
        
        key, *rest = line.split("=", 1)
        key = key.strip()
        
        if key in whisper_configs:
            updated_lines.append(f"{key}={whisper_configs[key]}")
            config_found[key] = True
        else:
            updated_lines.append(line)
    
    # 添加未找到的配置项
    for key, found in config_found.items():
        if not found:
            updated_lines.append(f"{key}={whisper_configs[key]}")
    
    # 写回.env文件
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(updated_lines))
    
    print(f".env文件已更新: {env_file}")

if __name__ == "__main__":
    main() 