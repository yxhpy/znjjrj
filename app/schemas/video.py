from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Union
from datetime import datetime
from app.core.database import VideoType, TaskStatus
from enum import Enum

class VideoUrlRequest(BaseModel):
    url: HttpUrl
    video_type: VideoType = Field(..., description="视频类型：movie(电影)或course(教学视频)")
    
class VideoUploadResponse(BaseModel):
    task_id: str
    message: str = "视频上传成功，开始处理"
    status: TaskStatus

class TaskCreate(BaseModel):
    """创建任务请求模型"""
    video_path: Optional[str] = None
    video_url: Optional[str] = None
    video_type: VideoType = VideoType.MOVIE
    use_omni: bool = False  # 是否使用全模态模型

class TaskResponse(BaseModel):
    """任务创建响应模型"""
    task_id: str
    message: str

class TaskStatusResponse(BaseModel):
    """任务状态响应模型"""
    id: str
    status: str
    progress: float
    message: Optional[str] = None
    output_path: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    use_omni: Optional[bool] = False

class ChapterPoint(BaseModel):
    """章节时间点"""
    time: float  # 时间点（秒）
    title: str   # 章节标题

class VideoAnalysisResult(BaseModel):
    """视频分析结果"""
    video_type: VideoType
    duration: float
    chapter_points: List[ChapterPoint] = []
    transcript: str  # 原视频文字记录
    narration_script: str  # 生成的解说脚本 