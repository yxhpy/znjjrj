"""
语音合成(TTS)服务模块，使用策略模式实现不同TTS引擎的支持
"""
import os
import logging
from abc import ABC, abstractmethod
from app.config import settings

logger = logging.getLogger(__name__)

class TTSBase(ABC):
    """TTS服务抽象基类"""
    
    @abstractmethod
    def synthesize(self, text: str, output_path: str) -> str:
        """
        将文本转换为语音
        
        Args:
            text (str): 要合成的文本内容
            output_path (str): 输出音频文件路径
            
        Returns:
            str: 输出音频文件路径
        """
        pass


class GTTSService(TTSBase):
    """Google Text-to-Speech服务实现"""
    
    def __init__(self):
        self.lang = settings.GTTS_LANG
        self.slow = settings.GTTS_SLOW
        logger.info(f"初始化GTTSService，语言：{self.lang}，慢速：{self.slow}")
    
    def synthesize(self, text: str, output_path: str) -> str:
        try:
            # 使用gTTS进行文本到语音转换
            from gtts import gTTS
            
            tts = gTTS(text=text, lang=self.lang, slow=self.slow)
            tts.save(output_path)
            logger.info(f"gTTS语音合成成功：{output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"gTTS语音合成失败: {str(e)}")
            raise


class CoquiTTSService(TTSBase):
    """Coqui TTS服务实现"""
    
    def __init__(self):
        self.model_name = settings.COQUI_MODEL_NAME
        self.vocoder_name = settings.COQUI_VOCODER_NAME  # 保留这个属性以兼容配置，但不再使用
        self.use_gpu = settings.COQUI_USE_GPU
        self.tts = None
        logger.info(f"初始化CoquiTTSService，模型：{self.model_name}，GPU：{self.use_gpu}")
    
    def _load_model(self):
        """加载Coqui TTS模型"""
        try:
            import torch
            from TTS.api import TTS
            
            # 检查GPU可用性
            device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
            
            # 使用新版本的API初始化TTS
            # 注意：新版本不再需要单独指定vocoder_name，而是直接使用模型路径
            self.tts = TTS(self.model_name).to(device)
            logger.info(f"Coqui TTS模型加载成功，使用设备：{device}")
        except ImportError:
            logger.error("无法导入TTS包，请确保已安装：pip install TTS")
            raise
        except Exception as e:
            logger.error(f"加载Coqui TTS模型失败: {str(e)}")
            raise
    
    def synthesize(self, text: str, output_path: str) -> str:
        try:
            # 延迟加载模型，只在首次使用时加载
            if self.tts is None:
                self._load_model()
            
            # 合成语音
            self.tts.tts_to_file(text=text, file_path=output_path)
            logger.info(f"Coqui TTS语音合成成功：{output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Coqui TTS语音合成失败: {str(e)}")
            raise


class ChatTTSService(TTSBase):
    """ChatTTS服务实现（基于API的TTS服务）"""
    
    def __init__(self):
        self.api_key = settings.CHATTTS_API_KEY
        self.voice = settings.CHATTTS_VOICE
        self.rate = settings.CHATTTS_RATE
        self.pitch = settings.CHATTTS_PITCH
        logger.info(f"初始化ChatTTSService，声音：{self.voice}，语速：{self.rate}，音调：{self.pitch}")
    
    def synthesize(self, text: str, output_path: str) -> str:
        try:
            import requests
            
            # 检查API密钥是否配置
            if not self.api_key:
                raise ValueError("ChatTTS API密钥未配置，请在.env文件中设置CHATTTS_API_KEY")
            
            # 发送请求到ChatTTS API（这里使用一个假设的API端点，实际应根据ChatTTS的文档进行适配）
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "voice": self.voice,
                "rate": self.rate,
                "pitch": self.pitch,
                "output_format": "mp3"
            }
            
            response = requests.post(
                "https://api.chattts.com/v1/synthesize",  # 假设的API端点
                headers=headers,
                json=data
            )
            
            if response.status_code != 200:
                raise Exception(f"ChatTTS API请求失败: {response.status_code} {response.text}")
            
            # 保存音频文件
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            logger.info(f"ChatTTS语音合成成功：{output_path}")
            return output_path
        except Exception as e:
            logger.error(f"ChatTTS语音合成失败: {str(e)}")
            raise


class TTSFactory:
    """TTS服务工厂类，根据配置创建适当的TTS服务实例"""
    
    @staticmethod
    def create_tts_service() -> TTSBase:
        """
        根据配置创建TTS服务实例
        
        Returns:
            TTSBase: TTS服务实例
        """
        tts_type = settings.TTS_SERVICE_TYPE.lower()
        
        if tts_type == "gtts":
            return GTTSService()
        elif tts_type == "coqui":
            return CoquiTTSService()
        elif tts_type == "chattts":
            return ChatTTSService()
        else:
            logger.warning(f"未知的TTS服务类型: {tts_type}，将使用默认的gTTS服务")
            return GTTSService()


# 单例模式实现，确保TTS服务只被初始化一次
_tts_service_instance = None

def get_tts_service() -> TTSBase:
    """
    获取TTS服务实例（单例模式）
    
    Returns:
        TTSBase: TTS服务实例
    """
    global _tts_service_instance
    
    if _tts_service_instance is None:
        _tts_service_instance = TTSFactory.create_tts_service()
    
    return _tts_service_instance 