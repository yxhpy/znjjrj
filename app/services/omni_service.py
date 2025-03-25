import os
import base64
import logging
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import uuid

import requests
from openai import OpenAI
import numpy as np
from moviepy.editor import VideoFileClip

from app.config import settings, TEMP_DIR
from app.services.frame_extraction import create_frame_extractor

logger = logging.getLogger(__name__)

class OmniService:
    """阿里云全模态模型服务，用于视频内容理解"""
    
    def __init__(self):
        """初始化全模态模型服务"""
        # 从环境变量获取配置或使用默认值
        self.api_key = os.getenv("ALIYUN_API_KEY", "")
        self.base_url = os.getenv("ALIYUN_OMNI_BASE_URL", "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation")
        self.model_name = os.getenv("ALIYUN_OMNI_MODEL", "qwen-omni-turbo")
        
        if not self.api_key:
            logger.warning("阿里云API密钥未配置，请设置ALIYUN_API_KEY环境变量")
    
    def _encode_video_base64(self, video_path: str, max_duration: int = 60) -> str:
        """
        将视频编码为base64字符串
        
        Args:
            video_path: 视频文件路径
            max_duration: 最大处理时长(秒)，超过将截取
        
        Returns:
            base64编码的视频字符串
        """
        try:
            # 首先尝试使用ffmpeg直接处理，避免moviepy内存问题
            try:
                import subprocess
                import tempfile
                import os
                
                # 检查视频时长
                duration_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', video_path
                ]
                duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
                logger.info(f"视频时长: {duration}秒")
                
                # 如果视频过长，截取处理
                if duration > max_duration:
                    logger.info(f"视频时长({duration}秒)超过限制({max_duration}秒)，进行截取")
                    
                    # 创建临时文件存储截取后的视频
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # 使用ffmpeg截取视频
                    ffmpeg_cmd = [
                        'ffmpeg', '-i', video_path, '-t', str(max_duration),
                        '-c:v', 'libx264', '-crf', '28', '-preset', 'fast',
                        '-an', '-y', temp_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 使用截取后的视频
                    video_to_encode = temp_path
                else:
                    video_to_encode = video_path
                
                # 读取视频文件并进行base64编码
                with open(video_to_encode, "rb") as video_file:
                    encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
                
                # 如果创建了临时文件则删除
                if duration > max_duration:
                    os.unlink(temp_path)
                    logger.info(f"已删除临时截取视频: {temp_path}")
                
                logger.info(f"成功编码视频为base64，大小: {len(encoded_video)} 字符")
                return encoded_video
                
            except Exception as ffmpeg_error:
                logger.error(f"使用ffmpeg处理视频失败: {str(ffmpeg_error)}")
                logger.info("尝试使用moviepy作为备选方案...")
            
            # 如果ffmpeg方法失败，使用moviepy作为备选方案
            # 检查是否需要截取视频
            clip = VideoFileClip(video_path)
            duration = clip.duration
            
            if duration > max_duration:
                logger.info(f"视频时长({duration}秒)超过限制({max_duration}秒)，进行截取")
                # 创建临时文件存储截取后的视频
                temp_file = TEMP_DIR / f"trimmed_{uuid.uuid4()}.mp4"
                # 截取视频
                trimmed_clip = clip.subclip(0, max_duration)
                trimmed_clip.write_videofile(str(temp_file), codec="libx264")
                # 关闭原始视频
                clip.close()
                trimmed_clip.close()
                # 使用截取后的视频
                video_path = str(temp_file)
            else:
                clip.close()
            
            # 读取视频文件并进行base64编码
            with open(video_path, "rb") as video_file:
                encoded_video = base64.b64encode(video_file.read()).decode('utf-8')
            
            # 如果创建了临时文件则删除
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
                
            return encoded_video
        except Exception as e:
            logger.error(f"视频编码失败: {str(e)}")
            raise
    
    def _extract_frames(self, video_path: str, num_frames: int = 5) -> List[str]:
        """
        从视频中提取关键帧并编码为base64
        
        Args:
            video_path: 视频文件路径
            num_frames: 要提取的帧数量
        
        Returns:
            List[str]: 帧的base64编码列表
        """
        try:
            # 使用ffmpeg提取帧，避免moviepy的内存问题
            try:
                import subprocess
                import tempfile
                import glob
                import os
                
                logger.info(f"使用ffmpeg提取 {num_frames} 帧...")
                
                # 获取视频时长
                duration_cmd = [
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1', video_path
                ]
                duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
                logger.info(f"视频时长: {duration}秒")
                
                # 创建临时目录存储帧图像
                with tempfile.TemporaryDirectory() as temp_dir:
                    # 计算时间间隔
                    interval = duration / (num_frames + 1)
                    frames_base64 = []
                    
                    # 提取每个时间点的帧
                    for i in range(1, num_frames + 1):
                        time_point = interval * i
                        output_file = os.path.join(temp_dir, f"frame_{i:03d}.jpg")
                        
                        # 使用ffmpeg提取帧
                        ffmpeg_cmd = [
                            'ffmpeg', '-i', video_path, '-ss', str(time_point),
                            '-frames:v', '1', '-q:v', '2', '-y', output_file
                        ]
                        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        # 检查文件是否创建成功
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            # 读取图片并编码为base64
                            with open(output_file, 'rb') as f:
                                img_data = f.read()
                            img_str = base64.b64encode(img_data).decode('utf-8')
                            frames_base64.append(img_str)
                            logger.info(f"成功提取时间点 {time_point:.2f}s 的帧")
                        else:
                            logger.error(f"提取时间点 {time_point:.2f}s 的帧失败")
                    
                    if frames_base64:
                        logger.info(f"成功提取 {len(frames_base64)} 帧")
                        return frames_base64
                    else:
                        logger.warning("未能提取任何帧，尝试提取视频首帧")
                        # 尝试提取首帧
                        output_file = os.path.join(temp_dir, "first_frame.jpg")
                        ffmpeg_cmd = [
                            'ffmpeg', '-i', video_path, '-ss', '0',
                            '-frames:v', '1', '-q:v', '2', '-y', output_file
                        ]
                        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        
                        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                            with open(output_file, 'rb') as f:
                                img_data = f.read()
                            img_str = base64.b64encode(img_data).decode('utf-8')
                            return [img_str]
                        
                        raise Exception("无法提取任何帧")
            except Exception as ffmpeg_error:
                logger.error(f"使用ffmpeg提取帧失败: {str(ffmpeg_error)}")
                # 如果ffmpeg失败，尝试使用moviepy作为备选方案
                logger.info("尝试使用moviepy作为备选方案...")
            
            # 以下是moviepy备选方案代码
            clip = VideoFileClip(video_path)
            duration = clip.duration
            logger.info(f"视频时长: {duration}秒, 分辨率: {clip.size}")
            
            # 计算提取帧的时间点
            timestamps = [i * duration / (num_frames + 1) for i in range(1, num_frames + 1)]
            
            # 提取帧并编码
            frames_base64 = []
            for ts in timestamps:
                try:
                    # 获取指定时间的帧
                    frame = clip.get_frame(ts)
                    logger.info(f"帧形状: {frame.shape}, 帧类型: {frame.dtype}")
                    
                    # 将帧转换为PIL图像，然后转为base64
                    from PIL import Image
                    from io import BytesIO
                    
                    # 检查帧的形状并适当处理
                    if len(frame.shape) == 3:
                        if frame.shape[0] == 3:  # 通道在第一维 (3, H, W)
                            logger.info("帧格式为 (通道,高,宽)，进行转置")
                            frame = np.transpose(frame, (1, 2, 0))
                        elif frame.shape[2] != 3:  # 非标准RGB格式
                            logger.info(f"非标准RGB格式，形状为 {frame.shape}，尝试转换")
                            if frame.shape[2] == 4:  # RGBA格式
                                frame = frame[:, :, :3]  # 去掉Alpha通道
                    else:
                        logger.error(f"无法处理的帧形状: {frame.shape}")
                        continue
                    
                    # 确保数值范围在0-255之间
                    if frame.dtype != np.uint8:
                        logger.info(f"转换帧数据类型从 {frame.dtype} 到 uint8")
                        if frame.max() <= 1.0:  # 如果值在0-1之间
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    
                    # 转换为PIL图像
                    pil_image = Image.fromarray(frame)
                    buffer = BytesIO()
                    pil_image.save(buffer, format="JPEG")
                    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    frames_base64.append(img_str)
                    logger.info(f"成功提取时间点 {ts} 的帧")
                except Exception as frame_error:
                    logger.error(f"处理时间点 {ts} 的帧时出错: {str(frame_error)}")
                    # 继续处理其他帧
            
            # 关闭视频
            clip.close()
            
            # 确保至少返回一个帧
            if not frames_base64 and num_frames > 0:
                logger.warning("未能提取任何有效帧，尝试使用替代方法")
                try:
                    # 使用替代方法 - 直接使用视频的第一帧图像文件
                    import subprocess
                    import tempfile
                    
                    # 创建临时文件存储帧
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    # 使用ffmpeg提取第一帧
                    ffmpeg_cmd = [
                        'ffmpeg', '-i', video_path, 
                        '-vframes', '1', '-an', '-ss', '0', 
                        '-y', temp_path
                    ]
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                    
                    # 读取提取的帧
                    with open(temp_path, 'rb') as f:
                        img_data = f.read()
                    
                    # 编码为base64
                    img_str = base64.b64encode(img_data).decode('utf-8')
                    frames_base64.append(img_str)
                    
                    # 清理临时文件
                    os.unlink(temp_path)
                    logger.info("成功使用ffmpeg提取视频第一帧")
                except Exception as alt_error:
                    logger.error(f"替代方法提取帧失败: {str(alt_error)}")
            
            logger.info(f"成功提取 {len(frames_base64)} 帧")
            return frames_base64
        except Exception as e:
            logger.error(f"提取视频帧失败: {str(e)}")
            raise
    
    def analyze_video_content(self, video_path: str, task_description: str = None) -> Dict[str, Any]:
        """
        分析视频内容，提取关键信息
        
        Args:
            video_path: 视频文件路径
            task_description: 任务描述，指导模型如何分析视频
        
        Returns:
            Dict: 视频内容分析结果
        """
        try:
            if not self.api_key:
                raise ValueError("阿里云API密钥未配置，请设置ALIYUN_API_KEY环境变量")
            
            # 默认任务描述
            if task_description is None:
                task_description = """
                请详细分析这段视频的内容，包括:
                1. 视频中出现的主要画面内容
                2. 画面中的关键对象和活动
                3. 视频的主题和目的
                4. 视频中的关键时刻
                5. 如果要剪辑这个视频，哪些部分是最重要的
                
                请以结构化的JSON格式回答，包含以下字段:
                {
                    "摘要": "视频内容概述",
                    "关键对象": ["对象1", "对象2", "..."],
                    "关键时刻": [
                        {"时间点": "xx秒", "描述": "发生了什么"},
                        {"时间点": "xx秒", "描述": "发生了什么"}
                    ],
                    "重要片段": [
                        {"开始": "xx秒", "结束": "xx秒", "内容": "这个片段的重要内容"},
                        {"开始": "xx秒", "结束": "xx秒", "内容": "这个片段的重要内容"}
                    ],
                    "剪辑建议": "关于如何剪辑的建议"
                }
                """
            
            # 安全地获取视频信息
            try:
                video_clip = VideoFileClip(video_path)
                duration = video_clip.duration
                width, height = video_clip.size
                fps = video_clip.fps
                video_clip.close()
                logger.info(f"视频信息: 时长={duration}秒, 分辨率={width}x{height}, FPS={fps}")
            except Exception as video_info_error:
                logger.error(f"获取视频信息失败: {str(video_info_error)}")
                # 使用ffprobe作为备选方案
                try:
                    import subprocess
                    # 避免这里使用json作为变量名
                    import json as json_module
                    
                    cmd = [
                        'ffprobe', '-v', 'error', '-select_streams', 'v:0', 
                        '-show_entries', 'stream=width,height,duration,r_frame_rate', 
                        '-of', 'json', video_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    info = json_module.loads(result.stdout)
                    
                    stream = info.get('streams', [{}])[0]
                    width = int(stream.get('width', 320))
                    height = int(stream.get('height', 240))
                    duration = float(stream.get('duration', 5))
                    
                    # 解析帧率 (通常是分数形式，如"24/1")
                    fps_str = stream.get('r_frame_rate', '24/1')
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        fps = num / den if den != 0 else 24
                    else:
                        fps = float(fps_str)
                    
                    logger.info(f"使用ffprobe获取视频信息: 时长={duration}秒, 分辨率={width}x{height}, FPS={fps}")
                except Exception as ffprobe_error:
                    logger.error(f"使用ffprobe获取视频信息失败: {str(ffprobe_error)}")
                    # 使用保守的默认值
                    duration = 5
                    width, height = 320, 240
                    fps = 24
                    logger.warning(f"无法获取视频信息，使用默认值: 时长={duration}秒, 分辨率={width}x{height}, FPS={fps}")
            
            # 创建OpenAI客户端（使用百炼兼容模式）
            client = OpenAI(
                api_key=self.api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            
            # 根据视频时长和大小决定使用什么策略
            if duration > 60 or (width * height * 3 * fps * duration) > 100_000_000:  # 判断视频是否过大
                # 长视频或大视频使用关键帧分析策略
                logger.info(f"视频较大(时长={duration}秒, 分辨率={width}x{height})，使用关键帧分析策略")
                frames = self._extract_frames(video_path, num_frames=min(10, int(duration / 6) + 1))
                
                if not frames:
                    logger.error("无法提取视频帧，无法进行分析")
                    return {"error": "无法提取视频帧", "suggestions": "请检查视频文件格式或使用其他视频"}
                
                # 构建请求消息 - 按照新的API格式
                system_message = {
                    "role": "system", 
                    "content": [{"type": "text", "text": "你是一个专业的视频分析专家，善于从关键帧中分析视频内容。请使用中文回复。"}]
                }
                
                user_content = [
                    {"type": "text", "text": f"这是一个时长为{duration}秒，分辨率为{width}x{height}的视频的关键帧。{task_description}"}
                ]
                
                # 添加关键帧作为图像
                for i, frame in enumerate(frames):
                    user_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                    })
                
                user_message = {"role": "user", "content": user_content}
                
                messages = [system_message, user_message]
            else:
                # 短视频直接使用视频分析
                logger.info(f"视频较小(时长={duration}秒, 分辨率={width}x{height})，直接进行视频分析")
                try:
                    video_base64 = self._encode_video_base64(video_path)
                    
                    # 按照新的API格式构建消息
                    system_message = {
                        "role": "system", 
                        "content": [{"type": "text", "text": "你是一个专业的视频分析专家，善于理解视频内容。请使用中文回复。"}]
                    }
                    
                    user_content = [
                        {"type": "video_url", "video_url": {"url": f"data:;base64,{video_base64}"}},
                        {"type": "text", "text": task_description}
                    ]
                    
                    user_message = {"role": "user", "content": user_content}
                    
                    messages = [system_message, user_message]
                except Exception as encode_error:
                    logger.error(f"视频编码失败，改用关键帧分析: {str(encode_error)}")
                    # 如果视频编码失败，转而使用关键帧
                    frames = self._extract_frames(video_path, num_frames=min(10, int(duration / 6) + 1))
                    
                    if not frames:
                        logger.error("无法提取视频帧，无法进行分析")
                        return {"error": "无法提取视频帧", "suggestions": "请检查视频文件格式或使用其他视频"}
                    
                    # 构建请求消息 - 按照新的API格式
                    system_message = {
                        "role": "system", 
                        "content": [{"type": "text", "text": "你是一个专业的视频分析专家，善于从关键帧中分析视频内容。请使用中文回复。"}]
                    }
                    
                    user_content = [
                        {"type": "text", "text": f"这是一个时长为{duration}秒，分辨率为{width}x{height}的视频的关键帧。{task_description}"}
                    ]
                    
                    # 添加关键帧作为图像
                    for i, frame in enumerate(frames):
                        user_content.append({
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{frame}"}
                        })
                    
                    user_message = {"role": "user", "content": user_content}
                    
                    messages = [system_message, user_message]
            
            # 使用OpenAI客户端发送请求，与示例代码保持一致
            try:
                # 发送API请求前记录关键信息
                if isinstance(messages, list) and len(messages) > 0:
                    logger.info(f"准备发送API请求到{self.base_url}, 模型: {self.model_name}")
                    if len(messages) > 1 and 'content' in messages[1] and isinstance(messages[1]['content'], list):
                        content_types = [item.get('type', 'unknown') for item in messages[1]['content'] if isinstance(item, dict)]
                        logger.info(f"请求包含的内容类型: {content_types}")
                
                # 发送API请求
                logger.info("开始调用全模态模型API...")
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    stream=True,
                )
                
                # 处理流式响应
                full_response = ""
                usage_info = None
                
                logger.info("开始接收API响应流...")
                chunk_count = 0
                for chunk in completion:
                    chunk_count += 1
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        if chunk_count % 10 == 0:  # 每10个块记录一次，避免日志过多
                            logger.debug(f"已接收{chunk_count}个响应块，当前响应长度: {len(full_response)} 字符")
                
                logger.info(f"API响应接收完成，总共接收{chunk_count}个块，响应总长度: {len(full_response)} 字符")
                
                # 尝试将完整响应解析为JSON
                if full_response:
                    try:
                        logger.info("尝试将响应解析为JSON...")
                        analysis_result = json.loads(full_response)
                        logger.info(f"成功解析为JSON，包含的字段: {list(analysis_result.keys())}")
                    except json.JSONDecodeError:
                        # 如果不是有效的JSON，尝试提取JSON部分
                        logger.warning("响应不是有效的JSON格式，尝试提取JSON部分...")
                        import re
                        json_pattern = r'```json\s*([\s\S]*?)\s*```'
                        json_match = re.search(json_pattern, full_response)
                        
                        if json_match:
                            # 提取JSON字符串并解析
                            json_str = json_match.group(1)
                            logger.info(f"找到JSON部分，长度为{len(json_str)}字符")
                            try:
                                analysis_result = json.loads(json_str)
                                logger.info(f"成功解析提取的JSON，包含的字段: {list(analysis_result.keys())}")
                            except json.JSONDecodeError:
                                logger.error("提取的JSON部分解析失败")
                                analysis_result = {"raw_response": full_response}
                        else:
                            # 如果无法提取，返回原始响应
                            logger.warning("无法从响应中提取JSON部分，返回原始响应")
                            analysis_result = {"raw_response": full_response}
                else:
                    logger.error("API未返回任何内容")
                    analysis_result = {"error": "未收到有效响应"}
                
                return analysis_result
                
            except Exception as api_error:
                logger.error(f"API调用失败: {str(api_error)}")
                return {"error": "API调用失败", "details": str(api_error)}
            
        except Exception as e:
            logger.error(f"视频内容分析失败: {str(e)}")
            raise
    
    def suggest_video_edits(self, video_path: str, transcript: str = None) -> Dict[str, Any]:
        """
        基于视频内容分析，提供剪辑建议
        
        Args:
            video_path: 视频文件路径
            transcript: 视频转录文本(可选)
            
        Returns:
            Dict: 视频剪辑建议
        """
        try:
            # 制定剪辑任务描述
            task_description = """
            请分析这段视频内容，作为专业视频剪辑师，提供详细的剪辑建议。
            
            请关注以下方面:
            1. 识别视频中最吸引人、最有价值的部分
            2. 确定哪些部分可以被剪掉以提高视频质量和节奏
            3. 指出视频的关键时刻和转场点
            4. 提供精确的时间戳，标注建议保留的重要片段
            5. 分析视频节奏，并建议如何提高观看体验
            
            请以结构化的JSON格式回答，包含以下字段:
            - keep_segments: 建议保留的片段列表，每个包含开始和结束时间点以及重要性说明
            - remove_segments: 建议删除的片段列表，每个包含开始和结束时间点以及原因
            - key_moments: 关键时刻列表及其时间点
            - pacing_suggestions: 节奏调整建议
            - overall_strategy: 整体剪辑策略
            """
            
            # 如果提供了转录文本，添加到请求中
            if transcript:
                task_description += f"\n\n以下是视频的文字转录内容，请结合画面和文字进行分析:\n{transcript}"
            
            # 调用视频分析方法
            editing_suggestions = self.analyze_video_content(video_path, task_description)
            
            return editing_suggestions
        except Exception as e:
            logger.error(f"生成视频剪辑建议失败: {str(e)}")
            raise
    
    def generate_summary_with_context(self, video_path: str, transcript: str = None, duration: float = None) -> Dict[str, Any]:
        """
        基于视频内容生成带有上下文的摘要，用于AI生成解说词
        
        Args:
            video_path: 视频文件路径
            transcript: 视频转录文本(可选)
            duration: 视频时长(可选)
            
        Returns:
            Dict: 包含摘要和上下文信息的结果
        """
        try:
            # 如果未提供时长，获取视频时长
            if duration is None:
                clip = VideoFileClip(video_path)
                duration = clip.duration
                clip.close()
            
            # 构建任务描述
            task_description = f"""
            请分析这段视频内容，提供详细且有深度的分析，包括视觉内容和上下文理解。
            
            视频时长为{duration}秒。
            
            请关注以下方面:
            1. 视频中出现的主要场景和元素
            2. 视觉叙事的流程和发展
            3. 画面中传达的情感和氛围
            4. 视频的主题和核心信息
            5. 视频中的象征意义和隐含信息
            
            请以结构化的JSON格式回答，包含以下字段:
            - detailed_summary: 详细的视频内容概述
            - visual_elements: 主要视觉元素分析
            - narrative_flow: 叙事流程分析
            - themes: 核心主题分析
            - context: 背景和上下文分析
            - emotional_tone: 情感基调分析
            """
            
            # 如果提供了转录文本，添加到请求中
            if transcript:
                task_description += f"\n\n以下是视频的文字转录内容，请结合视觉内容和文字进行全面分析:\n{transcript}"
            
            # 调用视频分析方法
            context_analysis = self.analyze_video_content(video_path, task_description)
            
            return context_analysis
        except Exception as e:
            logger.error(f"生成视频内容摘要失败: {str(e)}")
            raise

# 创建单例实例
omni_service = OmniService() 