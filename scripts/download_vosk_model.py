#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载Vosk中文语音识别模型
"""

import os
import sys
import shutil
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODELS_DIR

# Vosk中文模型下载链接
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-cn-0.22.zip"
MODEL_DIR = MODELS_DIR / "vosk-model-cn-0.22"
MODEL_ZIP = MODELS_DIR / "vosk-model-cn-0.22.zip"

def download_file(url, local_path):
    """
    下载文件，显示进度条
    """
    print(f"正在下载 {url} 到 {local_path}...")
    
    # 创建父目录
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 开始下载
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(local_path, 'wb') as f, tqdm(
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

def extract_zip(zip_path, extract_to):
    """
    解压ZIP文件
    """
    print(f"正在解压 {zip_path} 到 {extract_to}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # 获取总解压大小
        total_size = sum(file.file_size for file in zip_ref.infolist())
        
        # 解压文件并显示进度
        extracted_size = 0
        with tqdm(desc="解压进度", total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
            for file in zip_ref.infolist():
                zip_ref.extract(file, extract_to)
                extracted_size += file.file_size
                bar.update(file.file_size)

def main():
    """
    主函数
    """
    # 检查模型是否已下载
    if MODEL_DIR.exists():
        print(f"模型已存在: {MODEL_DIR}")
        return
    
    try:
        # 下载模型
        # download_file(MODEL_URL, MODEL_ZIP)
        
        # 解压模型
        extract_zip(MODEL_ZIP, MODELS_DIR)
        
        # 删除ZIP文件
        os.remove(MODEL_ZIP)
        
        print(f"Vosk中文模型下载并安装成功: {MODEL_DIR}")
        
    except Exception as e:
        print(f"下载或解压失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 