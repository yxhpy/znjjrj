from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import enum
from datetime import datetime

from app.config import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)

# 创建会话
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建基础模型类
Base = declarative_base()

# 定义任务状态枚举
class TaskStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# 定义视频类型枚举
class VideoType(str, enum.Enum):
    MOVIE = "movie"  # 电影
    COURSE = "course"  # 教学视频

# 任务模型
class Task(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, index=True)
    video_path = Column(String(255))  # 视频文件路径
    video_url = Column(String(255), nullable=True)  # 视频URL，如果有
    video_type = Column(Enum(VideoType))  # 视频类型
    status = Column(Enum(TaskStatus), default=TaskStatus.PENDING)
    progress = Column(Float, default=0.0)  # 进度（0-100）
    message = Column(Text, nullable=True)  # 状态消息
    result_path = Column(String(255), nullable=True)  # 结果视频路径
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# 初始化数据库
def init_db():
    Base.metadata.create_all(bind=engine)

# 获取数据库会话
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 