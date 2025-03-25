import os
from pydantic import BaseSettings
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
TEMP_DIR = BASE_DIR / "temp"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
MODELS_DIR = BASE_DIR / "models"  # 添加模型目录

# 确保目录存在
for dir_path in [TEMP_DIR, OUTPUT_DIR, UPLOAD_DIR, MODELS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

load_dotenv()


class Settings(BaseSettings):
    # 应用设置
    APP_NAME: str = "视频自动剪辑解说API"
    API_V1_STR: str = "/api/v1"

    # 大模型设置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "sk-aa7d81b5035649ccb5d14b2e7940b73c")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_MODEL_NAME: str = os.getenv("MODEL_NAME", os.getenv("OPENAI_MODEL_NAME", "gpt-4-turbo"))

    # 模型提供商类型 (openai, azure, local, dashscope等)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")

    # 大模型处理设置
    TRANSCRIPT_CHUNK_SIZE: int = int(os.getenv("TRANSCRIPT_CHUNK_SIZE", "8000"))  # 文本分片大小（字符数）

    # 语音识别服务设置
    SPEECH_RECOGNITION_SERVICE_TYPE: str = os.getenv("SPEECH_RECOGNITION_SERVICE_TYPE", "baidu")

    # 百度语音识别设置
    BAIDU_APP_ID: str = os.getenv("BAIDU_APP_ID", "")
    BAIDU_API_KEY: str = os.getenv("BAIDU_API_KEY", "")
    BAIDU_SECRET_KEY: str = os.getenv("BAIDU_SECRET_KEY", "")

    # 开源语音识别设置 (Vosk)
    VOSK_MODEL_PATH: str = os.getenv("VOSK_MODEL_PATH", str(MODELS_DIR / "vosk-model-cn-0.22"))

    # Whisper语音识别设置
    WHISPER_MODEL_PATH: str = os.getenv("WHISPER_MODEL_PATH", "")
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "base")
    WHISPER_CPP_MODEL_PATH: str = os.getenv("WHISPER_CPP_MODEL_PATH", str(MODELS_DIR / "whisper.cpp" / "ggml-base.bin"))
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"

    # TTS服务设置
    TTS_SERVICE_TYPE: str = os.getenv("TTS_SERVICE_TYPE", "gtts")

    # gTTS设置
    GTTS_LANG: str = os.getenv("GTTS_LANG", "zh-cn")
    GTTS_SLOW: bool = os.getenv("GTTS_SLOW", "False").lower() == "true"

    # Coqui TTS设置
    COQUI_MODEL_NAME: str = os.getenv("COQUI_MODEL_NAME", "tts_models/zh-CN/baker/tacotron2-DDC-GST")
    COQUI_VOCODER_NAME: str = os.getenv("COQUI_VOCODER_NAME", "vocoder_models/universal/libri-tts/fullband-melgan")
    COQUI_USE_GPU: bool = os.getenv("COQUI_USE_GPU", "False").lower() == "true"

    # ChatTTS设置
    CHATTTS_API_KEY: str = os.getenv("CHATTTS_API_KEY", "")
    CHATTTS_VOICE: str = os.getenv("CHATTTS_VOICE", "default")
    CHATTTS_RATE: float = float(os.getenv("CHATTTS_RATE", "1.0"))
    CHATTTS_PITCH: float = float(os.getenv("CHATTTS_PITCH", "1.0"))

    # 视频设置
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024 * 1024  # 10 GB
    SUPPORTED_FORMATS: list = ["mp4", "avi", "mov", "mkv", "flv", "webm"]

    # 任务设置
    TASK_EXPIRY_DAYS: int = 7  # 任务结果保存天数

    # 数据库设置
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/video_api.db"

    # 日志设置
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
