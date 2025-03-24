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
        
        try:
            # 默认选项
            default_options = {
                "language": "zh",  # 默认中文
                "beam_size": 5,    # 波束搜索大小
                "task": "transcribe"  # 转录任务
            }
            
            # 合并用户提供的选项
            if options:
                default_options.update(options)
                
            # 根据使用的库调用不同的处理方法
            if hasattr(self, 'WhisperModel') and self.WhisperModel is not None:
                return self._process_with_faster_whisper(audio_data, audio_format, sample_rate, default_options)
            elif hasattr(self, 'whisper_cpp') and self.whisper_cpp is not None:
                return self._process_with_whisper_cpp(audio_data, audio_format, sample_rate, default_options)
            else:
                return "[无可用的Whisper实现]"
                
        except Exception as e:
            logger.error(f"Whisper语音识别服务错误: {str(e)}")
            return f"[识别错误: {str(e)}]"
    
    def _process_with_faster_whisper(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any]) -> str:
        """使用faster-whisper处理音频"""
        import io
        import soundfile as sf
        
        # 将字节数据转换为音频数组
        audio_io = io.BytesIO(audio_data)
        audio_array, _ = sf.read(audio_io)
        
        # 识别
        segments, info = self.model.transcribe(
            audio_array, 
            beam_size=options.get("beam_size", 5),
            language=options.get("language", "zh"),
            task=options.get("task", "transcribe"),
            initial_prompt=options.get("initial_prompt", None)
        )
        
        # 合并结果
        result = " ".join([segment.text for segment in segments])
        
        return result.strip()
    
    def _process_with_whisper_cpp(self, audio_data: bytes, audio_format: str, sample_rate: int, options: Dict[str, Any]) -> str:
        """使用whisper.cpp处理音频"""
        import io
        import wave
        import numpy as np
        
        # 将字节数据转换为PCM格式
        with io.BytesIO(audio_data) as audio_io:
            with wave.open(audio_io, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_array = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 设置语言和选项
        whisper_params = {
            "language": options.get("language", "zh"),
            "n_threads": options.get("n_threads", 4),
            "translate": options.get("task", "transcribe") == "translate",
        }
        
        # 识别
        result = self.model.transcribe(audio_array, whisper_params)
        
        return result.strip()


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