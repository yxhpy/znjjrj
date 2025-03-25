#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
阿里云全模态模型服务测试脚本

测试OmniService的功能，包括视频编码、帧提取和内容分析
"""

import os
import sys
import logging
import tempfile
from pathlib import Path
import json
import base64
import argparse
from unittest.mock import patch, MagicMock
import subprocess

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from io import BytesIO

from app.config import TEMP_DIR, UPLOAD_DIR
from app.services.omni_service import OmniService, omni_service

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_video(duration=2, width=160, height=120, fps=10):
    """创建测试视频文件"""
    # 完全使用ffmpeg创建测试视频，避开moviepy可能导致的reshape问题
    
    # 确保TEMP_DIR存在
    os.makedirs(TEMP_DIR, exist_ok=True)
    test_video_path = os.path.join(TEMP_DIR, "test_omni_video.mp4")
    
    # 创建红色、绿色、蓝色的固态色彩视频片段
    try:
        # 设置每个颜色部分的时长
        part_duration = duration / 3
        
        # 创建临时文件保存三个颜色片段
        with tempfile.TemporaryDirectory() as temp_dir:
            red_video = os.path.join(temp_dir, "red.mp4")
            green_video = os.path.join(temp_dir, "green.mp4")
            blue_video = os.path.join(temp_dir, "blue.mp4")
            concat_file = os.path.join(temp_dir, "concat.txt")
            
            # 创建红色视频
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', 
                '-i', f'color=c=red:s={width}x{height}:d={part_duration}:r={fps}',
                '-c:v', 'libx264', '-crf', '32', '-preset', 'ultrafast',
                red_video
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 创建绿色视频
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', 
                '-i', f'color=c=green:s={width}x{height}:d={part_duration}:r={fps}',
                '-c:v', 'libx264', '-crf', '32', '-preset', 'ultrafast',
                green_video
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 创建蓝色视频
            subprocess.run([
                'ffmpeg', '-y', '-f', 'lavfi', 
                '-i', f'color=c=blue:s={width}x{height}:d={part_duration}:r={fps}',
                '-c:v', 'libx264', '-crf', '32', '-preset', 'ultrafast',
                blue_video
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 创建concat文件
            with open(concat_file, 'w') as f:
                f.write(f"file '{red_video}'\n")
                f.write(f"file '{green_video}'\n")
                f.write(f"file '{blue_video}'\n")
            
            # 连接视频片段
            subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
                '-i', concat_file, '-c', 'copy',
                test_video_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            logger.info(f"使用ffmpeg创建测试视频: {test_video_path}, 时长: {duration}秒, 分辨率: {width}x{height}")
    except Exception as e:
        logger.error(f"使用ffmpeg创建测试视频失败: {e}")
        # 如果ffmpeg方法失败，使用超级简单的备选方案创建单帧视频
        try:
            # 创建一个单帧的红色图像
            red_frame = np.zeros((height, width, 3), dtype=np.uint8)
            red_frame[:, :, 0] = 255  # 红色
            
            # 保存为图像
            temp_img = os.path.join(TEMP_DIR, "red_frame.jpg")
            Image.fromarray(red_frame).save(temp_img)
            
            # 使用ffmpeg将图像转为视频
            subprocess.run([
                'ffmpeg', '-y', '-loop', '1', '-i', temp_img, 
                '-t', str(duration), '-vcodec', 'libx264', 
                '-pix_fmt', 'yuv420p', '-r', str(fps),
                test_video_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 清理临时图像
            if os.path.exists(temp_img):
                os.remove(temp_img)
                
            logger.info(f"使用备选方案创建测试视频: {test_video_path}, 时长: {duration}秒, 分辨率: {width}x{height}")
        except Exception as alt_e:
            logger.error(f"备选方案创建测试视频也失败: {alt_e}")
            raise
    
    return test_video_path

def test_video_base64_encoding():
    """测试视频Base64编码"""
    logger.info("测试视频Base64编码...")
    
    try:
        # 创建测试视频
        test_video_path = create_test_video(duration=2)  # 较短的视频
        
        # 创建服务实例
        service = OmniService()
        
        # 测试编码
        encoded_video = service._encode_video_base64(test_video_path)
        
        # 验证编码结果
        assert encoded_video and isinstance(encoded_video, str), "编码结果应为非空字符串"
        
        # 检查是否为有效的base64
        try:
            decoded = base64.b64decode(encoded_video)
            assert len(decoded) > 0, "解码后应有数据"
            logger.info(f"编码成功，base64长度: {len(encoded_video)}, 解码后大小: {len(decoded)} 字节")
        except Exception as e:
            logger.error(f"解码base64失败: {str(e)}")
            assert False, "应产生有效的base64编码"
        
        return True
    except Exception as e:
        logger.error(f"视频编码测试失败: {str(e)}")
        return False
    finally:
        # 清理
        if 'test_video_path' in locals() and os.path.exists(test_video_path):
            try:
                os.remove(test_video_path)
                logger.info(f"已删除测试视频: {test_video_path}")
            except:
                pass

def test_frame_extraction():
    """测试视频帧提取"""
    logger.info("测试视频帧提取...")
    
    try:
        # 创建测试视频
        test_video_path = create_test_video(duration=2)
        
        # 创建服务实例
        service = OmniService()
        
        # 测试帧提取
        num_frames = 3
        frames = service._extract_frames(test_video_path, num_frames)
        
        # 验证结果
        assert frames and isinstance(frames, list), "应返回帧列表"
        assert len(frames) <= num_frames, f"应提取不超过 {num_frames} 帧"
        
        # 验证每个帧
        for i, frame in enumerate(frames):
            # 检查是否为有效的base64
            try:
                decoded = base64.b64decode(frame)
                # 尝试打开为图像
                img = Image.open(BytesIO(decoded))
                assert img.width > 0 and img.height > 0, "应为有效图像"
                logger.info(f"帧 {i+1}: 大小 {img.width}x{img.height}, 格式 {img.format}")
            except Exception as e:
                logger.error(f"验证帧 {i+1} 失败: {str(e)}")
                assert False, f"帧 {i+1} 应为有效图像"
        
        logger.info(f"成功提取 {len(frames)} 帧")
        return True
    except Exception as e:
        logger.error(f"帧提取测试失败: {str(e)}")
        return False
    finally:
        # 清理
        if 'test_video_path' in locals() and os.path.exists(test_video_path):
            try:
                os.remove(test_video_path)
                logger.info(f"已删除测试视频: {test_video_path}")
            except:
                pass

def test_video_analysis_with_mock():
    """使用模拟响应测试视频分析功能"""
    logger.info("使用模拟响应测试视频分析...")
    
    try:
        # 创建测试视频
        test_video_path = create_test_video(duration=2)
        
        # 创建模拟响应
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "output": {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({
                                "summary": "这是一个测试视频，包含红、绿、蓝三个颜色部分",
                                "key_objects": ["红色画面", "绿色画面", "蓝色画面"],
                                "key_moments": [
                                    {"time": 0.0, "description": "红色部分开始"},
                                    {"time": 0.7, "description": "绿色部分开始"},
                                    {"time": 1.4, "description": "蓝色部分开始"}
                                ]
                            })
                        }
                    }
                ]
            }
        }
        
        # 首先确保视频文件的基本信息正确，避免moviepy读取错误
        try:
            # 使用ffprobe获取视频信息
            duration_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', test_video_path
            ]
            duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
            
            info_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'json', test_video_path
            ]
            info_output = subprocess.check_output(info_cmd).decode('utf-8')
            video_info = json.loads(info_output)
            width = video_info['streams'][0]['width']
            height = video_info['streams'][0]['height']
            
            logger.info(f"测试视频信息: 时长={duration}s, 分辨率={width}x{height}")
        except Exception as probe_error:
            logger.warning(f"无法检索视频信息: {probe_error}")
        
        # 模拟requests.post方法
        with patch('requests.post', return_value=mock_response):
            # 创建服务实例
            service = OmniService()
            service.api_key = "mock_api_key"  # 确保不会因为缺少API密钥而跳过测试
            
            # 模拟ffmpeg方法避免内部调用moviepy
            with patch('subprocess.run', return_value=MagicMock()):
                with patch('subprocess.check_output', return_value=b"2.0"):
                    # 测试视频分析
                    result = service.analyze_video_content(test_video_path)
            
            # 验证结果
            assert isinstance(result, dict), "应返回字典结果"
            assert "summary" in result, "结果应包含摘要"
            assert "key_objects" in result, "结果应包含关键对象"
            assert "key_moments" in result, "结果应包含关键时刻"
            
            logger.info(f"视频分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"视频分析测试失败: {str(e)}")
        return False
    finally:
        # 清理
        if 'test_video_path' in locals() and os.path.exists(test_video_path):
            try:
                os.remove(test_video_path)
                logger.info(f"已删除测试视频: {test_video_path}")
            except:
                pass

def test_real_api_call(video_path=None):
    """测试真实API调用（需要配置API密钥）"""
    if not video_path:
        logger.info("未提供视频路径，跳过真实API调用测试")
        return False
    
    logger.info(f"使用真实API测试视频分析: {video_path}")
    
    try:
        # 验证文件存在
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False
        
        # 获取环境变量中的API密钥
        api_key = os.getenv("ALIYUN_API_KEY")
        if not api_key:
            logger.error("未设置ALIYUN_API_KEY环境变量，跳过真实API调用测试")
            return False
        
        # 首先确保视频文件的基本信息正确
        try:
            # 使用ffprobe获取视频信息
            duration_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ]
            duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
            
            info_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate',
                '-of', 'json', video_path
            ]
            info_output = subprocess.check_output(info_cmd).decode('utf-8')
            video_info = json.loads(info_output)
            width = video_info['streams'][0]['width']
            height = video_info['streams'][0]['height']
            
            logger.info(f"真实视频信息: 时长={duration}s, 分辨率={width}x{height}")
        except Exception as probe_error:
            logger.warning(f"无法检索视频信息: {probe_error}")
        
        # 创建服务实例并设置API密钥
        service = OmniService()
        service.api_key = api_key
        
        # 测试视频分析
        result = service.analyze_video_content(video_path)
        
        # 验证结果
        assert isinstance(result, dict), "应返回字典结果"
        
        # 输出结果
        logger.info(f"视频分析结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
        return True
    except Exception as e:
        logger.error(f"真实API调用测试失败: {str(e)}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="测试阿里云全模态模型服务")
    parser.add_argument("--video", type=str, help="用于真实API测试的视频文件路径")
    parser.add_argument("--test-real-api", action="store_true", help="是否执行真实API调用测试")
    parser.add_argument("--only-real-api", action="store_true", help="只执行真实API调用测试，跳过其他测试")
    parser.add_argument("--skip-base64", action="store_true", help="跳过视频Base64编码测试")
    parser.add_argument("--skip-frames", action="store_true", help="跳过视频帧提取测试")
    parser.add_argument("--skip-mock", action="store_true", help="跳过视频分析模拟测试")
    args = parser.parse_args()
    
    logger.info("开始OmniService测试...")
    
    # 如果指定只运行真实API测试，则设置跳过其他测试
    if args.only_real_api:
        args.skip_base64 = True
        args.skip_frames = True
        args.skip_mock = True
        args.test_real_api = True
        
        if not args.video:
            logger.error("使用--only-real-api参数时必须提供--video参数指定视频文件路径")
            return
    
    # 测试视频编码
    if not args.skip_base64:
        try:
            if test_video_base64_encoding():
                logger.info("视频Base64编码测试通过")
            else:
                logger.error("视频Base64编码测试失败")
        except Exception as e:
            logger.error(f"执行视频Base64编码测试时发生未捕获异常: {str(e)}")
    else:
        logger.info("跳过视频Base64编码测试")
    
    # 测试帧提取
    if not args.skip_frames:
        try:
            if test_frame_extraction():
                logger.info("视频帧提取测试通过")
            else:
                logger.error("视频帧提取测试失败")
        except Exception as e:
            logger.error(f"执行视频帧提取测试时发生未捕获异常: {str(e)}")
    else:
        logger.info("跳过视频帧提取测试")
    
    # 测试带模拟的视频分析
    if not args.skip_mock:
        try:
            if test_video_analysis_with_mock():
                logger.info("视频分析模拟测试通过")
            else:
                logger.error("视频分析模拟测试失败")
        except Exception as e:
            logger.error(f"执行视频分析模拟测试时发生未捕获异常: {str(e)}")
    else:
        logger.info("跳过视频分析模拟测试")
    
    # 测试真实API调用
    if args.test_real_api:
        video_path = args.video if args.video else None
        try:
            if test_real_api_call(video_path):
                logger.info("真实API调用测试通过")
            else:
                logger.warning("真实API调用测试跳过或失败")
        except Exception as e:
            logger.error(f"执行真实API调用测试时发生未捕获异常: {str(e)}")
    else:
        logger.info("跳过真实API调用测试，使用 --test-real-api 参数启用")
    
if __name__ == "__main__":
    main() 