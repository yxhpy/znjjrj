from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Enum, Boolean
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
    QUEUED = "queued"

# 定义视频类型枚举
class VideoType(str, enum.Enum):
    MOVIE = "movie"  # 电影
    COURSE = "course"  # 教学视频

# 任务模型
class Task(Base):
    """处理任务模型"""
    __tablename__ = "tasks"
    
    id = Column(String, primary_key=True)
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.QUEUED)
    video_type = Column(Enum(VideoType), nullable=False, default=VideoType.MOVIE)
    video_path = Column(String)
    video_url = Column(String)
    result_path = Column(String)  # 原来的output_path
    progress = Column(Float, default=0.0)
    message = Column(String)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime)  # 原来的completed_at
    
    # 添加use_omni字段
    use_omni = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            "id": self.id,
            "status": self.status.value,
            "video_type": self.video_type.value,
            "video_path": self.video_path,
            "video_url": self.video_url,
            "output_path": self.result_path,  # 为了保持API兼容性
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.updated_at.isoformat() if self.updated_at else None,
            "use_omni": self.use_omni
        }

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