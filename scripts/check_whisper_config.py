#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查和修复Whisper语音识别所需的配置
"""

import os
import sys
import importlib
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def check_config_file():
    """
    检查config.py文件中是否包含Whisper相关配置
    """
    print("检查配置文件中的Whisper设置...")
    
    try:
        # 尝试导入config模块
        from app.config import settings
        
        # 检查Whisper所需的配置
        required_attrs = [
            "WHISPER_MODEL_SIZE", 
            "WHISPER_MODEL_PATH", 
            "WHISPER_CPP_MODEL_PATH",
            "USE_GPU"
        ]
        
        missing_attrs = []
        for attr in required_attrs:
            if not hasattr(settings, attr):
                missing_attrs.append(attr)
        
        if missing_attrs:
            print(f"警告: 配置中缺少以下Whisper相关设置: {', '.join(missing_attrs)}")
            return False
        else:
            print("配置文件中包含所有必要的Whisper设置")
            return True
            
    except ImportError as e:
        print(f"错误: 无法导入配置模块: {e}")
        return False
    except Exception as e:
        print(f"检查配置文件时出错: {e}")
        return False

def update_config_file():
    """
    更新config.py文件以添加缺失的Whisper配置
    """
    print("正在更新config.py文件以添加Whisper配置...")
    
    config_path = Path(__file__).resolve().parent.parent / "app" / "config.py"
    
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return False
    
    try:
        # 读取config.py内容
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # 检查是否已包含Whisper配置
        if "WHISPER_MODEL_SIZE" in content:
            print("配置文件已包含Whisper设置，无需修改")
            return True
        
        # 查找Settings类定义
        if "class Settings(BaseSettings):" not in content:
            print("错误: 无法找到Settings类定义")
            return False
        
        # 找到开源语音识别设置部分
        vosk_setting_pos = content.find("# 开源语音识别设置")
        if vosk_setting_pos == -1:
            print("警告: 无法找到开源语音识别设置部分，将尝试查找其他插入点")
            # 尝试找到视频设置部分
            insert_pos = content.find("# 视频设置")
            if insert_pos == -1:
                print("错误: 无法找到合适的插入点")
                return False
        else:
            # 找到Vosk设置行的结尾
            vosk_line_end = content.find("\n", content.find("VOSK_MODEL_PATH", vosk_setting_pos))
            if vosk_line_end == -1:
                print("错误: 无法确定VOSK_MODEL_PATH设置的结束位置")
                return False
            insert_pos = vosk_line_end + 1
        
        # 构建Whisper配置
        whisper_config = """
    # Whisper语音识别设置
    WHISPER_MODEL_PATH: str = os.getenv("WHISPER_MODEL_PATH", "")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_CPP_MODEL_PATH: str = os.getenv("WHISPER_CPP_MODEL_PATH", str(MODELS_DIR / "whisper.cpp" / "ggml-base.bin"))
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"
    """
        
        # 插入Whisper配置
        new_content = content[:insert_pos] + whisper_config + content[insert_pos:]
        
        # 写回文件
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"配置文件已更新: {config_path}")
        
        # 重新加载config模块
        if "app.config" in sys.modules:
            importlib.reload(sys.modules["app.config"])
        
        return True
        
    except Exception as e:
        print(f"更新配置文件时出错: {e}")
        return False

def check_whisper_installation():
    """
    检查Whisper依赖安装状态
    """
    print("检查Whisper依赖安装状态...")
    
    missing_packages = []
    
    # 检查faster-whisper
    try:
        import faster_whisper
        print("✓ faster-whisper 已安装")
    except ImportError:
        print("✗ faster-whisper 未安装")
        missing_packages.append("faster-whisper")
    
    # 检查whisper.cpp Python绑定
    try:
        import whisper_cpp
        print("✓ whisper-cpp-python 已安装")
    except ImportError:
        print("✗ whisper-cpp-python 未安装")
        missing_packages.append("whisper-cpp-python")
    
    # 检查soundfile
    try:
        import soundfile
        print("✓ soundfile 已安装")
    except ImportError:
        print("✗ soundfile 未安装")
        missing_packages.append("soundfile")
    
    return missing_packages

def main():
    """
    主函数
    """
    print("检查Whisper语音识别配置...\n")
    
    # 检查配置文件
    config_ok = check_config_file()
    if not config_ok:
        print("\n尝试更新配置文件...")
        if update_config_file():
            print("配置文件更新成功")
        else:
            print("无法自动更新配置文件，请手动修改app/config.py")
    
    # 检查依赖安装
    print("\n检查Whisper依赖...")
    missing_packages = check_whisper_installation()
    
    if missing_packages:
        print(f"\n缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
    
    # 检查模型文件
    print("\n检查Whisper模型文件...")
    
    from app.config import MODELS_DIR, settings
    
    # 检查faster-whisper模型
    whisper_model_ok = False
    
    # 检查whisper.cpp模型
    cpp_model_path = Path(settings.WHISPER_CPP_MODEL_PATH)
    if cpp_model_path.exists():
        print(f"✓ Whisper.cpp模型已存在: {cpp_model_path}")
        whisper_model_ok = True
    else:
        print(f"✗ Whisper.cpp模型不存在: {cpp_model_path}")
    
    # 检查是否需要下载模型
    if not whisper_model_ok:
        print("\n需要下载Whisper模型文件")
        print("请运行以下命令下载:")
        print(f"python scripts/download_whisper_model.py --model-size {settings.WHISPER_MODEL_SIZE}")
    
    print("\n检查完成。")

if __name__ == "__main__":
    main() 