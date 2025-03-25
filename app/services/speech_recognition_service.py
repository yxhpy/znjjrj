import os
import logging
import wave
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any

from app.config import settings

logger = logging.getLogger(__name__)

class SpeechRecognitionService(ABC):
    """语音识别服务抽象基类"""
    
    @abstractmethod
    def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any] = None) -> str:
        """
        识别音频数据并返回文本
        
        Args:
            audio_data: 音频数据字节
            audio_format: 音频格式 (wav, mp3等)
            sample_rate: 采样率
            options: 附加选项
            
        Returns:
            str: 识别出的文本
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""
        pass


class BaiduSpeechRecognitionService(SpeechRecognitionService):
    """百度语音识别服务实现"""
    
    def __init__(self):
        # 导入百度语音识别SDK
        try:
            from aip import AipSpeech
            self.AipSpeech = AipSpeech
            
            # 百度语音识别配置
            self.app_id = settings.BAIDU_APP_ID
            self.api_key = settings.BAIDU_API_KEY
            self.secret_key = settings.BAIDU_SECRET_KEY
            
            # 初始化客户端
            if self.is_available():
                self.client = self.AipSpeech(self.app_id, self.api_key, self.secret_key)
            else:
                self.client = None
                
        except ImportError:
            logger.error("未安装百度语音识别SDK，请安装: pip install baidu-aip")
            self.AipSpeech = None
            self.client = None
    
    def is_available(self) -> bool:
        """检查百度语音识别服务是否可用"""
        if not hasattr(self, 'AipSpeech') or self.AipSpeech is None:
            return False
        return all([self.app_id, self.api_key, self.secret_key])
    
    def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any] = None) -> str:
        """使用百度语音识别服务识别音频"""
        if not self.is_available():
            logger.error("百度语音识别服务不可用")
            return "[百度语音识别服务不可用]"
        
        try:
            # 百度语音识别参数
            asr_options = {
                'dev_pid': 1537,  # 普通话(支持简单的英文识别)
                'format': audio_format,  # 音频格式
                'rate': sample_rate,  # 采样率
                'channel': 1,  # 声道数
            }
            
            # 合并用户提供的选项
            if options:
                asr_options.update(options)
            
            result = self.client.asr(audio_data, audio_format, sample_rate, asr_options)
            
            # 检查识别结果
            if result['err_no'] == 0 and 'result' in result:
                return result['result'][0]
            else:
                logger.warning(f"百度语音识别失败: {result.get('err_msg', '未知错误')}")
                return "[语音识别失败]"
                
        except Exception as e:
            logger.error(f"百度语音识别服务请求错误: {str(e)}")
            return "[识别错误]"


class OpenSourceSpeechRecognitionService(SpeechRecognitionService):
    """开源语音识别服务实现 (使用Vosk)"""
    
    def __init__(self):
        # 尝试导入Vosk库
        try:
            import vosk
            self.vosk = vosk
            
            # 检查模型文件是否存在
            self.model_path = settings.VOSK_MODEL_PATH
            if not os.path.exists(self.model_path):
                logger.error(f"Vosk模型文件不存在: {self.model_path}")
                self.model = None
            else:
                # 加载模型
                self.model = self.vosk.Model(self.model_path)
                
        except ImportError:
            logger.error("未安装Vosk库，请安装: pip install vosk")
            self.vosk = None
            self.model = None
    
    def is_available(self) -> bool:
        """检查开源语音识别服务是否可用"""
        if not hasattr(self, 'vosk') or self.vosk is None:
            return False
        return self.model is not None
    
    def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any] = None) -> str:
        """使用Vosk进行语音识别"""
        if not self.is_available():
            logger.error("开源语音识别服务不可用")
            return "[开源语音识别服务不可用]"
        
        try:
            # 创建识别器
            recognizer = self.vosk.KaldiRecognizer(self.model, sample_rate)
            
            # 开始识别
            recognizer.AcceptWaveform(audio_data)
            result_json = recognizer.FinalResult()
            
            # 解析结果
            import json
            result = json.loads(result_json)
            
            if 'text' in result:
                return result['text']
            else:
                return "[无法识别]"
                
        except Exception as e:
            logger.error(f"开源语音识别服务错误: {str(e)}")
            return "[识别错误]"


class WhisperSpeechRecognitionService(SpeechRecognitionService):
    """Whisper加速版本语音识别服务实现"""
    
    def __init__(self, model_size: str = "base"):
        # 尝试导入faster-whisper库
        try:
            from faster_whisper import WhisperModel
            self.WhisperModel = WhisperModel
            
            # Whisper模型配置
            self.model_size = model_size
            self.model_path = settings.WHISPER_MODEL_PATH
            
            # 如果指定了本地模型路径且存在，则使用本地模型
            if self.model_path and os.path.exists(self.model_path):
                self.model = self.WhisperModel(self.model_path)
                logger.info(f"使用本地Whisper模型: {self.model_path}")
            else:
                # 否则使用预训练模型
                self.model = self.WhisperModel(self.model_size, device="cuda" if settings.USE_GPU else "cpu", compute_type="float16" if settings.USE_GPU else "int8", local_files_only=False)
                logger.info(f"使用预训练Whisper模型: {self.model_size}")
                
        except ImportError:
            logger.error("未安装faster-whisper库，请安装: pip install faster-whisper")
            self.WhisperModel = None
            self.model = None
            
        # 尝试加载替代实现: whisper.cpp的Python绑定
        if self.model is None:
            try:
                import whisper_cpp
                self.whisper_cpp = whisper_cpp
                self.model = whisper_cpp.Whisper(settings.WHISPER_CPP_MODEL_PATH)
                logger.info(f"使用whisper.cpp模型: {settings.WHISPER_CPP_MODEL_PATH}")
            except ImportError:
                logger.error("未安装whisper_cpp库，请安装: pip install whisper-cpp-python")
                self.whisper_cpp = None
                self.model = None
    
    def is_available(self) -> bool:
        """检查Whisper语音识别服务是否可用"""
        return self.model is not None
    
    def recognize(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any] = None) -> str:
        """使用Whisper进行语音识别"""
        if not self.is_available():
            logger.error("Whisper语音识别服务不可用")
            return "[Whisper语音识别服务不可用]"
        
        # 初始化临时文件路径变量
        temp_path = None
        converted_path = None
        
        try:
            # 验证音频数据
            if not audio_data or len(audio_data) < 100:  # 音频数据太小
                logger.error("音频数据无效或太小")
                return "[音频数据无效]"
            
            # 检测音频格式
            detected_format = self._detect_audio_format(audio_data)
            if detected_format and detected_format != audio_format:
                logger.warning(f"音频格式可能不正确，声明格式: {audio_format}, 检测格式: {detected_format}")
                audio_format = detected_format
            
            # 检查音频是否包含实际内容（不只是静音或噪音）
            has_audio_content = False
            try:
                if audio_format.lower() == 'wav' and len(audio_data) > 44:
                    # 从WAV文件中提取PCM数据
                    pcm_data = audio_data[44:]  # 跳过WAV头
                    # 将字节转换为16位整数数组
                    import numpy as np
                    samples = np.frombuffer(pcm_data, dtype=np.int16)
                    # 计算音量
                    if len(samples) > 0:
                        volume = np.abs(samples).mean()
                        std_dev = np.std(samples)
                        logger.info(f"音频平均音量: {volume:.2f}, 标准差: {std_dev:.2f}")
                        # 使用音量和标准差来判断是否是静音
                        if volume > 50 or std_dev > 100:  # 调整这些阈值
                            has_audio_content = True
                        else:
                            logger.warning(f"音频可能只包含静音或低水平噪声 (音量={volume:.2f}, 标准差={std_dev:.2f})")
            except Exception as e:
                logger.warning(f"检查音频内容时出错: {str(e)}")
            
            # 强制继续处理，即使疑似无内容
            if not has_audio_content:
                logger.warning("音频可能没有实际内容，但仍将尝试识别")
            
            # 默认选项
            default_options = {
                "language": "zh",  # 默认中文
                "beam_size": 5,    # 波束搜索大小
                "task": "transcribe",  # 转录任务
                "initial_prompt": "这是一段可能包含语音的音频内容"  # 添加初始提示
            }
            
            # 合并用户提供的选项
            if options:
                default_options.update(options)
                
            # 验证并规范化音频格式
            audio_format = audio_format.lower().strip().replace(".", "")
            if audio_format not in ["wav", "mp3", "flac", "ogg", "m4a"]:
                logger.warning(f"不支持的音频格式: {audio_format}，尝试作为wav处理")
                audio_format = "wav"
                
            # 根据使用的库调用不同的处理方法
            result = ""
            if hasattr(self, 'WhisperModel') and self.WhisperModel is not None:
                result = self._process_with_faster_whisper(audio_data, audio_format, sample_rate, default_options)
            elif hasattr(self, 'whisper_cpp') and self.whisper_cpp is not None:
                result = self._process_with_whisper_cpp(audio_data, audio_format, sample_rate, default_options)
            else:
                result = "[无可用的Whisper实现]"
            
            # 添加详细日志，显示提取的台词内容
            if result and len(result) > 0:
                logger.info(f"成功提取台词，内容为: \n{result[:300]}{'...' if len(result) > 300 else ''}")
            else:
                logger.warning("未能提取出有效台词，结果为空")
                
            # 如果识别结果为空，尝试增加灵敏度并重新识别
            if not result.strip():
                logger.warning("第一次识别结果为空，尝试调整参数重新识别...")
                
                # 尝试分析音频是否有实际内容
                try:
                    cmd_analyze = ["ffmpeg", "-i", path_to_use, "-af", "volumedetect", "-f", "null", "/dev/null"]
                    analyze_result = subprocess.run(cmd_analyze, capture_output=True, text=True)
                    
                    # 提取均值音量
                    import re
                    mean_volume = None
                    for line in analyze_result.stderr.split('\n'):
                        match = re.search(r'mean_volume:\s*([-\d.]+)\s*dB', line)
                        if match:
                            mean_volume = float(match.group(1))
                            break
                    
                    if mean_volume is not None:
                        logger.info(f"检测到音频均值音量: {mean_volume} dB")
                        if mean_volume < -40:  # 非常低的音量
                            logger.warning(f"音频音量非常低 ({mean_volume} dB)，尝试使用音量提升")
                            # 创建音量提升版本
                            boosted_path = f"{path_to_use}.boosted.wav"
                            gain = min(40, abs(mean_volume) - 15)  # 保持一个合理的音量提升
                            
                            cmd_boost = ["ffmpeg", "-y", "-i", path_to_use, "-af", f"volume={gain}dB", boosted_path]
                            subprocess.run(cmd_boost, check=True, capture_output=True, text=True)
                            
                            if os.path.exists(boosted_path):
                                logger.info(f"已创建音量提升后的音频: {boosted_path} (增益: {gain}dB)")
                                # 使用提升后的音频再次尝试识别
                                segments, info = self.model.transcribe(
                                    boosted_path,
                                    beam_size=options.get("beam_size", 5),
                                    language=options.get("language", "zh"),
                                    task=options.get("task", "transcribe"),
                                    initial_prompt=options.get("initial_prompt", None),
                                    vad_filter=True,  # 添加语音活动检测过滤
                                    vad_parameters={"min_silence_duration_ms": 100}  # 降低静音检测阈值
                                )
                                
                                # 更新结果
                                result = " ".join([segment.text for segment in segments])
                                logger.info(f"音量提升后的识别结果长度: {len(result)}")
                                
                                # 清理临时文件
                                try:
                                    os.unlink(boosted_path)
                                except:
                                    pass
                except Exception as boost_error:
                    logger.warning(f"音量分析和提升失败: {str(boost_error)}")
                
                # 如果仍然没有识别结果，尝试更激进的参数
                if not result.strip():
                    logger.warning("音量提升后仍无识别结果，尝试更激进的参数...")
                    segments, info = self.model.transcribe(
                        path_to_use,
                        beam_size=options.get("beam_size", 5) * 2,  # 增加波束宽度
                        language=options.get("language", "zh"),
                        task=options.get("task", "transcribe"),
                        initial_prompt="这段音频可能包含低音量或不清晰的语音内容",
                        vad_filter=False,  # 禁用语音活动检测
                        temperature=0.7,  # 增加温度参数
                        best_of=5,  # 增加采样次数
                        condition_on_previous_text=False,  # 不依赖前文
                        no_speech_threshold=0.1  # 降低无语音检测阈值
                    )
                    
                    # 更新结果
                    result = " ".join([segment.text for segment in segments])
                    logger.info(f"使用激进参数后的识别结果长度: {len(result)}")
            
            return result.strip()
        except Exception as e:
            logger.error(f"Whisper语音识别服务错误: {str(e)}")
            return "[识别错误]"  # 返回错误信息而不是引发异常
        finally:
            # 不在这里清理临时文件，因为temp_path和converted_path可能未被正确初始化
            # 临时文件清理已经在各自的处理方法内部完成
            pass
    
    def _detect_audio_format(self, audio_data: bytes) -> str:
        """尝试检测音频数据的实际格式"""
        # 常见音频格式的文件头（魔数）
        signatures = {
            b'RIFF': 'wav',  # WAV格式
            b'ID3': 'mp3',   # MP3格式（带ID3标签）
            b'\xFF\xFB': 'mp3',  # MP3格式（不带ID3标签）
            b'fLaC': 'flac',  # FLAC格式
            b'OggS': 'ogg',   # OGG格式
        }
        
        # 检查前几个字节
        for signature, format_name in signatures.items():
            if audio_data.startswith(signature):
                return format_name
        
        # 无法检测格式
        return None
    
    def _process_with_faster_whisper(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any]) -> str:
        """使用faster-whisper处理音频"""
        import io
        import tempfile
        import os
        import subprocess
        from pydub import AudioSegment
        import pathlib
        import struct
        
        temp_path = None
        converted_path = None
        
        try:
            # 检查音频数据是否有效
            if len(audio_data) < 44:  # WAV头至少44字节
                logger.warning("音频数据太小，可能不是有效的音频文件")
                
                # 尝试生成一个有效的空WAV文件，然后追加原始数据作为音频样本
                try:
                    logger.info("尝试重新生成有效的WAV格式...")
                    # 创建一个带有有效WAV头的文件
                    valid_wav = self._generate_valid_wav(audio_data, sample_rate)
                    audio_data = valid_wav
                    audio_format = "wav"  # 强制设置为WAV格式
                except Exception as gen_error:
                    logger.warning(f"重新生成WAV文件失败: {str(gen_error)}")
            
            # 步骤1: 将原始音频数据保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                temp_path = os.path.abspath(temp_file.name)
                temp_file.write(audio_data)
                logger.info(f"原始音频已保存到临时文件: {temp_path}")
            
            # 检查文件类型
            try:
                file_type = subprocess.run(
                    ["file", "-b", temp_path], 
                    capture_output=True, 
                    text=True, 
                    check=True
                ).stdout.strip()
                logger.info(f"文件类型检测结果: {file_type}")
                
                if "audio" not in file_type.lower() and "wav" not in file_type.lower():
                    logger.warning(f"文件不是有效的音频格式: {file_type}")
                    # 尝试生成一个有效的WAV文件
                    rebuilt_path = f"{temp_path}.rebuilt.wav"
                    self._rebuild_wav_file(temp_path, rebuilt_path, sample_rate)
                    if os.path.exists(rebuilt_path):
                        temp_path = rebuilt_path
                        logger.info(f"已重建WAV文件: {temp_path}")
            except Exception as file_check_error:
                logger.warning(f"无法检查文件类型: {str(file_check_error)}")
            
            # 步骤2: 检查音频时长，对于过短的音频进行处理
            try:
                # 使用ffprobe检查音频时长
                cmd_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", temp_path]
                duration_result = subprocess.run(cmd_duration, capture_output=True, text=True)
                
                # 解析时长（秒）
                try:
                    duration = float(duration_result.stdout.strip())
                    logger.info(f"检测到音频时长: {duration:.4f}秒")
                    
                    # 如果时长小于1秒，则扩展音频
                    min_duration = 1.0  # 设置最小时长为1秒
                    if duration < min_duration:
                        logger.warning(f"音频太短 ({duration:.4f}秒 < {min_duration}秒)，需要扩展")
                        extended_path = f"{temp_path}.extended.wav"
                        
                        # 使用ffmpeg重复音频片段，直到达到最小时长
                        repeat_count = int(min_duration / duration) + 1
                        filter_complex = f"[0:a]aloop=loop={repeat_count}:size=32767[out]"
                        
                        cmd_extend = [
                            "ffmpeg", "-y", "-i", temp_path, 
                            "-filter_complex", filter_complex, 
                            "-map", "[out]", 
                            "-ar", "16000", "-ac", "1",
                            extended_path
                        ]
                        
                        subprocess.run(cmd_extend, check=True, capture_output=True, text=True)
                        
                        if os.path.exists(extended_path):
                            logger.info(f"已创建扩展音频文件: {extended_path}")
                            temp_path = extended_path
                        else:
                            logger.warning("扩展音频文件创建失败")
                except ValueError:
                    logger.warning(f"无法解析音频时长: {duration_result.stdout}")
            except Exception as duration_error:
                logger.warning(f"检查音频时长失败: {str(duration_error)}")
            
            # 步骤3: 使用ffmpeg直接转换为WAV格式（更可靠的方法）
            converted_path = f"{temp_path}.wav"
            try:
                # 首先检查音频格式，避免处理无效文件
                try:
                    cmd_info = ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "default=noprint_wrappers=1:nokey=1", temp_path]
                    format_info = subprocess.run(cmd_info, capture_output=True, text=True).stdout.strip()
                    logger.info(f"ffprobe检测到音频格式: {format_info}")
                except Exception as probe_error:
                    logger.warning(f"ffprobe检查失败: {str(probe_error)}")
                
                # 使用subprocess直接调用ffmpeg转换，确保使用绝对路径
                cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "0.1", converted_path]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"已创建一个空的有效WAV文件: {converted_path}")
                
                # 然后尝试将原始音频附加到这个有效的WAV文件
                cmd_append = ["ffmpeg", "-y", "-i", converted_path, "-i", temp_path, "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", "-ar", "16000", "-ac", "1", f"{converted_path}.fixed.wav"]
                subprocess.run(cmd_append, check=True, capture_output=True, text=True)
                
                if os.path.exists(f"{converted_path}.fixed.wav"):
                    converted_path = f"{converted_path}.fixed.wav"
                    logger.info(f"已创建修复后的WAV文件: {converted_path}")
                    path_to_use = converted_path
                else:
                    # 如果附加失败，则创建一个1秒的静音WAV文件作为替代
                    cmd_fallback = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "1", converted_path]
                    subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)
                    logger.warning(f"无法处理原始音频，使用静音文件替代: {converted_path}")
                    path_to_use = converted_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"音频转换失败，尝试直接使用原始文件: {str(e)}\n\nOutput from ffmpeg/avlib:\n\n{e.stdout}\n{e.stderr}")
                
                # 尝试创建一个静音的WAV文件
                try:
                    silence_path = f"{temp_path}_silence.wav"
                    cmd_silence = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "1", silence_path]
                    subprocess.run(cmd_silence, check=True, capture_output=True, text=True)
                    logger.info(f"已创建静音WAV文件作为替代: {silence_path}")
                    path_to_use = silence_path
                except Exception as silence_error:
                    logger.warning(f"创建静音文件也失败: {str(silence_error)}")
                    # 最后尝试读取一个系统测试音频文件或使用自生成的WAV
                    path_to_use = self._get_fallback_audio()
            
            # 验证音频文件是否可用
            try:
                with open(path_to_use, 'rb') as test_file:
                    test_data = test_file.read(1024)  # 读取部分数据测试
                    if len(test_data) < 44:  # WAV文件头至少44字节
                        raise ValueError("音频文件太小或无效")
                    
                    # 验证WAV头
                    if test_data.startswith(b'RIFF') and b'WAVE' in test_data[:44]:
                        logger.info(f"音频文件验证成功 (有效的WAV头): {path_to_use}")
                    else:
                        logger.warning(f"音频文件可能无效 (不是标准WAV格式): {path_to_use}")
                        # 尝试修复WAV头
                        fixed_path = self._fix_wav_header(path_to_use)
                        if fixed_path:
                            path_to_use = fixed_path
            except Exception as file_error:
                logger.error(f"音频文件验证失败: {str(file_error)}")
                # 使用静音文件作为最后的后备选项
                path_to_use = self._get_fallback_audio()
            
            # 步骤3: 使用文件路径进行音频识别
            logger.info(f"开始使用Whisper进行语音识别: {path_to_use}")
            
            # 检查路径并确保是绝对路径
            abs_path_to_use = os.path.abspath(path_to_use)
            if abs_path_to_use != path_to_use:
                logger.info(f"将相对路径转换为绝对路径: {path_to_use} -> {abs_path_to_use}")
                path_to_use = abs_path_to_use
                
            # 确认文件存在
            if not os.path.isfile(path_to_use):
                raise FileNotFoundError(f"音频文件不存在: {path_to_use}")
            
            # 进行最终验证
            try:
                with wave.open(path_to_use, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    framerate = wav_file.getframerate()
                    logger.info(f"WAV文件信息: 帧数={frames}, 采样宽度={sample_width}, 声道数={channels}, 采样率={framerate}")
                    
                    if frames == 0:
                        logger.warning("WAV文件不包含音频数据，使用静音文件")
                        path_to_use = self._get_fallback_audio()
            except Exception as wave_error:
                logger.warning(f"无法使用wave模块读取文件: {str(wave_error)}")
                # 尝试最后的修复方式
                path_to_use = self._get_fallback_audio()
                
            # 进行识别
            segments, info = self.model.transcribe(
                path_to_use,
                beam_size=options.get("beam_size", 5),
                language=options.get("language", "zh"),
                task=options.get("task", "transcribe"),
                initial_prompt=options.get("initial_prompt", None)
            )
            
            # 步骤4: 合并结果
            result = " ".join([segment.text for segment in segments])
            logger.info(f"语音识别完成，识别文本长度: {len(result)}")
            
            return result.strip()
        except Exception as e:
            logger.error(f"Whisper语音识别服务错误: {str(e)}")
            return "[识别错误]"  # 返回错误信息而不是引发异常
        finally:
            # 不在这里清理临时文件，因为temp_path和converted_path可能未被正确初始化
            # 临时文件清理已经在各自的处理方法内部完成
            pass
    
    def _generate_valid_wav(self, audio_data: bytes, sample_rate: int) -> bytes:
        """生成一个有效的WAV文件"""
        import struct
        import io
        
        # 创建WAV头
        channels = 1  # 单声道
        bits_per_sample = 16  # 16位采样
        
        # 如果没有有效的PCM数据，创建0.1秒的静音
        if len(audio_data) < 44 or not audio_data.startswith(b'RIFF'):
            # 每个采样2字节(16位)，静音采样值为0
            num_samples = int(0.1 * sample_rate)  # 0.1秒
            audio_samples = b'\x00\x00' * num_samples
        else:
            # 尝试提取原始音频数据(跳过头部)
            try:
                audio_samples = audio_data[44:]
            except:
                # 如果失败，创建静音
                num_samples = int(0.1 * sample_rate)
                audio_samples = b'\x00\x00' * num_samples
        
        # 计算数据大小
        data_size = len(audio_samples)
        file_size = 36 + data_size
        
        # 创建WAVE文件头
        wav_header = bytearray()
        wav_header.extend(b'RIFF')
        wav_header.extend(struct.pack('<I', file_size))
        wav_header.extend(b'WAVE')
        wav_header.extend(b'fmt ')
        wav_header.extend(struct.pack('<I', 16))  # fmt块大小
        wav_header.extend(struct.pack('<H', 1))   # 格式代码(PCM = 1)
        wav_header.extend(struct.pack('<H', channels))
        wav_header.extend(struct.pack('<I', sample_rate))
        wav_header.extend(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))  # 字节率
        wav_header.extend(struct.pack('<H', channels * bits_per_sample // 8))  # 块对齐
        wav_header.extend(struct.pack('<H', bits_per_sample))
        wav_header.extend(b'data')
        wav_header.extend(struct.pack('<I', data_size))
        
        # 拼接完整的WAV文件
        wav_file = bytearray()
        wav_file.extend(wav_header)
        wav_file.extend(audio_samples)
        
        return bytes(wav_file)
    
    def _rebuild_wav_file(self, input_path: str, output_path: str, sample_rate: int) -> bool:
        """尝试重新构建WAV文件"""
        try:
            # 读取原始文件数据
            with open(input_path, 'rb') as f:
                audio_data = f.read()
            
            # 生成有效的WAV文件
            wav_data = self._generate_valid_wav(audio_data, sample_rate)
            
            # 保存到输出路径
            with open(output_path, 'wb') as f:
                f.write(wav_data)
            
            return True
        except Exception as e:
            logger.warning(f"重新构建WAV文件失败: {str(e)}")
            return False
    
    def _fix_wav_header(self, file_path: str) -> str:
        """尝试修复WAV文件头"""
        try:
            # 读取原始文件数据
            with open(file_path, 'rb') as f:
                audio_data = f.read()
            
            # 生成修复后的文件路径
            fixed_path = f"{file_path}.fixed.wav"
            
            # 生成有效的WAV文件
            wav_data = self._generate_valid_wav(audio_data, 16000)
            
            # 保存到输出路径
            with open(fixed_path, 'wb') as f:
                f.write(wav_data)
            
            logger.info(f"WAV文件头已修复: {fixed_path}")
            return fixed_path
        except Exception as e:
            logger.warning(f"修复WAV文件头失败: {str(e)}")
            return None
    
    def _get_fallback_audio(self) -> str:
        """获取后备音频文件(静音WAV)"""
        try:
            # 创建临时文件
            import tempfile
            import subprocess
            import os
            
            fallback_path = os.path.join(tempfile.gettempdir(), f"whisper_fallback_{id(self)}.wav")
            
            # 使用ffmpeg创建1秒的静音
            cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "1", fallback_path]
            subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"已创建后备静音文件: {fallback_path}")
            return fallback_path
        except Exception as e:
            logger.warning(f"创建后备静音文件失败: {str(e)}")
            
            # 完全失败情况下，手动创建一个最小的有效WAV文件
            import tempfile
            fallback_path = os.path.join(tempfile.gettempdir(), f"whisper_fallback_manual_{id(self)}.wav")
            
            with open(fallback_path, 'wb') as f:
                f.write(self._generate_valid_wav(b'', 16000))
            
            logger.info(f"已手动创建最小WAV文件: {fallback_path}")
            return fallback_path

    def _process_with_whisper_cpp(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any]) -> str:
        """使用whisper.cpp处理音频"""
        import io
        import tempfile
        import os
        import subprocess
        import numpy as np
        from pydub import AudioSegment
        
        temp_path = None
        converted_path = None
        
        try:
            # 检查音频数据是否有效
            if len(audio_data) < 44:  # WAV头至少44字节
                logger.warning("音频数据太小，可能不是有效的音频文件")
                
                # 尝试生成一个有效的空WAV文件
                try:
                    logger.info("尝试重新生成有效的WAV格式...")
                    # 创建一个带有有效WAV头的文件
                    valid_wav = self._generate_valid_wav(audio_data, sample_rate)
                    audio_data = valid_wav
                    audio_format = "wav"  # 强制设置为WAV格式
                except Exception as gen_error:
                    logger.warning(f"重新生成WAV文件失败: {str(gen_error)}")
            
            # 步骤1: 将原始音频数据保存到临时文件
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as temp_file:
                temp_path = os.path.abspath(temp_file.name)
                temp_file.write(audio_data)
                logger.info(f"原始音频已保存到临时文件: {temp_path}")
            
            # 检查文件类型
            try:
                file_type = subprocess.run(
                    ["file", "-b", temp_path], 
                    capture_output=True, 
                    text=True, 
                    check=True
                ).stdout.strip()
                logger.info(f"文件类型检测结果: {file_type}")
                
                if "audio" not in file_type.lower() and "wav" not in file_type.lower():
                    logger.warning(f"文件不是有效的音频格式: {file_type}")
                    # 尝试生成一个有效的WAV文件
                    rebuilt_path = f"{temp_path}.rebuilt.wav"
                    self._rebuild_wav_file(temp_path, rebuilt_path, sample_rate)
                    if os.path.exists(rebuilt_path):
                        temp_path = rebuilt_path
                        logger.info(f"已重建WAV文件: {temp_path}")
            except Exception as file_check_error:
                logger.warning(f"无法检查文件类型: {str(file_check_error)}")
            
            # 步骤2: 检查音频时长，对于过短的音频进行处理
            try:
                # 使用ffprobe检查音频时长
                cmd_duration = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", temp_path]
                duration_result = subprocess.run(cmd_duration, capture_output=True, text=True)
                
                # 解析时长（秒）
                try:
                    duration = float(duration_result.stdout.strip())
                    logger.info(f"检测到音频时长: {duration:.4f}秒")
                    
                    # 如果时长小于1秒，则扩展音频
                    min_duration = 1.0  # 设置最小时长为1秒
                    if duration < min_duration:
                        logger.warning(f"音频太短 ({duration:.4f}秒 < {min_duration}秒)，需要扩展")
                        extended_path = f"{temp_path}.extended.wav"
                        
                        # 使用ffmpeg重复音频片段，直到达到最小时长
                        repeat_count = int(min_duration / duration) + 1
                        filter_complex = f"[0:a]aloop=loop={repeat_count}:size=32767[out]"
                        
                        cmd_extend = [
                            "ffmpeg", "-y", "-i", temp_path, 
                            "-filter_complex", filter_complex, 
                            "-map", "[out]", 
                            "-ar", "16000", "-ac", "1",
                            extended_path
                        ]
                        
                        subprocess.run(cmd_extend, check=True, capture_output=True, text=True)
                        
                        if os.path.exists(extended_path):
                            logger.info(f"已创建扩展音频文件: {extended_path}")
                            temp_path = extended_path
                        else:
                            logger.warning("扩展音频文件创建失败")
                except ValueError:
                    logger.warning(f"无法解析音频时长: {duration_result.stdout}")
            except Exception as duration_error:
                logger.warning(f"检查音频时长失败: {str(duration_error)}")
            
            # 步骤3: 使用ffmpeg直接转换为WAV格式（更可靠的方法）
            converted_path = f"{temp_path}.wav"
            try:
                # 首先检查音频格式
                try:
                    cmd_info = ["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "default=noprint_wrappers=1:nokey=1", temp_path]
                    format_info = subprocess.run(cmd_info, capture_output=True, text=True).stdout.strip()
                    logger.info(f"ffprobe检测到音频格式: {format_info}")
                except Exception as probe_error:
                    logger.warning(f"ffprobe检查失败: {str(probe_error)}")
                
                # 创建一个有效的WAV文件并尝试合并原始音频
                cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "0.1", converted_path]
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                logger.info(f"已创建一个空的有效WAV文件: {converted_path}")
                
                # 然后尝试将原始音频附加到这个有效的WAV文件
                cmd_append = ["ffmpeg", "-y", "-i", converted_path, "-i", temp_path, "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1", "-ar", "16000", "-ac", "1", f"{converted_path}.fixed.wav"]
                subprocess.run(cmd_append, check=True, capture_output=True, text=True)
                
                if os.path.exists(f"{converted_path}.fixed.wav"):
                    converted_path = f"{converted_path}.fixed.wav"
                    logger.info(f"已创建修复后的WAV文件: {converted_path}")
                    path_to_use = converted_path
                else:
                    # 如果附加失败，则创建一个1秒的静音WAV文件作为替代
                    cmd_fallback = ["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000", "-t", "1", converted_path]
                    subprocess.run(cmd_fallback, check=True, capture_output=True, text=True)
                    logger.warning(f"无法处理原始音频，使用静音文件替代: {converted_path}")
                    path_to_use = converted_path
            except subprocess.CalledProcessError as e:
                logger.warning(f"音频转换失败: {str(e)}\n\nOutput from ffmpeg/avlib:\n\n{e.stdout}\n{e.stderr}")
                # 使用静音文件作为后备
                path_to_use = self._get_fallback_audio()
            
            # 验证音频文件是否可用
            try:
                with open(path_to_use, 'rb') as test_file:
                    test_data = test_file.read(1024)  # 读取部分数据测试
                    if len(test_data) < 44:  # WAV文件头至少44字节
                        raise ValueError("音频文件太小或无效")
                    
                    # 验证WAV头
                    if test_data.startswith(b'RIFF') and b'WAVE' in test_data[:44]:
                        logger.info(f"音频文件验证成功 (有效的WAV头): {path_to_use}")
                    else:
                        logger.warning(f"音频文件可能无效 (不是标准WAV格式): {path_to_use}")
                        # 尝试修复WAV头
                        fixed_path = self._fix_wav_header(path_to_use)
                        if fixed_path:
                            path_to_use = fixed_path
            except Exception as file_error:
                logger.error(f"音频文件验证失败: {str(file_error)}")
                # 使用静音文件作为最后的后备选项
                path_to_use = self._get_fallback_audio()
            
            # 确保路径是绝对路径
            abs_path_to_use = os.path.abspath(path_to_use)
            if abs_path_to_use != path_to_use:
                logger.info(f"将相对路径转换为绝对路径: {path_to_use} -> {abs_path_to_use}")
                path_to_use = abs_path_to_use
            
            # 确认文件存在
            if not os.path.isfile(path_to_use):
                raise FileNotFoundError(f"音频文件不存在: {path_to_use}")
            
            # 步骤3: 读取并处理音频数据
            try:
                import wave
                with wave.open(path_to_use, 'rb') as wav_file:
                    frames = wav_file.readframes(wav_file.getnframes())
                    sample_width = wav_file.getsampwidth()
                    channels = wav_file.getnchannels()
                    framerate = wav_file.getframerate()
                    logger.info(f"WAV文件信息: 帧数={frames}, 采样宽度={sample_width}, 声道数={channels}, 采样率={framerate}")
                    
                    if frames == 0:
                        logger.warning("WAV文件不包含音频数据，使用静音文件")
                        path_to_use = self._get_fallback_audio()
                        with wave.open(path_to_use, 'rb') as silence_file:
                            frames = silence_file.readframes(silence_file.getnframes())
                            sample_width = silence_file.getsampwidth()
                    
                    # 根据采样宽度选择正确的数据类型
                    if sample_width == 1:
                        # 8-bit, 无符号
                        audio_array = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
                    elif sample_width == 2:
                        # 16-bit, 有符号
                        audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sample_width == 4:
                        # 32-bit, 有符号
                        audio_array = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        raise ValueError(f"不支持的采样宽度: {sample_width}")
                
                logger.info(f"使用wave模块成功读取音频数据，样本数: {len(audio_array)}")
            except Exception as wave_error:
                logger.warning(f"使用wave读取音频失败: {str(wave_error)}，尝试使用librosa")
                
                try:
                    import librosa
                    audio_array, _ = librosa.load(path_to_use, sr=16000, mono=True)
                    logger.info(f"使用librosa成功读取音频，样本数: {len(audio_array)}")
                except Exception as librosa_error:
                    logger.warning(f"librosa读取失败: {str(librosa_error)}，尝试使用soundfile")
                    try:
                        import soundfile as sf
                        audio_array, _ = sf.read(path_to_use)
                        if len(audio_array.shape) > 1:
                            audio_array = audio_array[:, 0]  # 取第一个声道
                        logger.info(f"使用soundfile成功读取音频，样本数: {len(audio_array)}")
                    except Exception as sf_error:
                        logger.warning(f"所有读取方法都失败，使用静音替代: {str(sf_error)}")
                        # 创建静音数据
                        audio_array = np.zeros(16000, dtype=np.float32)  # 1秒的静音
            
            # 步骤4: 设置语言和选项
            whisper_params = {
                "language": options.get("language", "zh"),
                "n_threads": options.get("n_threads", 4),
                "translate": options.get("task", "transcribe") == "translate",
            }
            
            # 步骤5: 识别
            logger.info(f"开始使用Whisper.cpp进行语音识别...")
            result = self.model.transcribe(audio_array, whisper_params)
            logger.info(f"语音识别完成，识别文本长度: {len(result)}")
            
            return result.strip()
        except Exception as e:
            logger.error(f"Whisper.cpp语音识别服务错误: {str(e)}")
            return "[识别错误]"  # 返回错误消息而不是抛出异常
        finally:
            # 不在这里清理临时文件，因为temp_path和converted_path可能未被正确初始化
            # 临时文件清理已经在各自的处理方法内部完成
            pass


class SpeechRecognitionFactory:
    """语音识别服务工厂"""
    
    @staticmethod
    def get_service(service_type: str = None) -> SpeechRecognitionService:
        """
        获取语音识别服务实例
        
        Args:
            service_type: 服务类型，可选值: 'baidu', 'opensource', 'whisper'，不指定则使用配置文件中的默认值
            
        Returns:
            SpeechRecognitionService: 语音识别服务实例
        """
        # 如果未指定类型，则使用配置中的默认值
        if service_type is None:
            service_type = settings.SPEECH_RECOGNITION_SERVICE_TYPE
        
        # 根据类型选择服务
        if service_type == 'baidu':
            service = BaiduSpeechRecognitionService()
        elif service_type == 'opensource':
            service = OpenSourceSpeechRecognitionService()
        elif service_type == 'whisper':
            service = WhisperSpeechRecognitionService(model_size=settings.WHISPER_MODEL_SIZE)
        else:
            logger.warning(f"未知的语音识别服务类型: {service_type}，将使用开源服务")
            service = OpenSourceSpeechRecognitionService()
        
        # 如果服务不可用，则尝试使用其他服务
        if not service.is_available():
            logger.warning(f"{service_type}语音识别服务不可用，尝试使用其他服务")
            
            # 尝试whisper
            fallback_service = WhisperSpeechRecognitionService()
            if fallback_service.is_available():
                return fallback_service
                
            # 尝试百度
            fallback_service = BaiduSpeechRecognitionService()
            if fallback_service.is_available():
                return fallback_service
                
            # 尝试开源
            fallback_service = OpenSourceSpeechRecognitionService()
            if fallback_service.is_available():
                return fallback_service
        
        return service 