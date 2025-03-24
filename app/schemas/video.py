from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from datetime import datetime
from app.core.database import VideoType, TaskStatus

class VideoUrlRequest(BaseModel):
    url: HttpUrl
    video_type: VideoType = Field(..., description="视频类型：movie(电影)或course(教学视频)")
    
class VideoUploadResponse(BaseModel):
    task_id: str
    message: str = "视频上传成功，开始处理"
    status: TaskStatus

class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    progress: float
    message: Optional[str] = None
    video_type: VideoType
    created_at: datetime
    updated_at: datetime
    result_path: Optional[str] = None
    
    class Config:
        orm_mode = True

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