import os
import shutil
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import uuid
import threading

from app.schemas.video import VideoUrlRequest, VideoUploadResponse, TaskStatusResponse, TaskCreate, TaskResponse
from app.core.database import get_db, VideoType, Task, TaskStatus
from app.config import UPLOAD_DIR, OUTPUT_DIR, settings
from app.services.task_service import task_service
from app.services.video_service import video_service

router = APIRouter()

@router.post("/upload", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_video(
    file: UploadFile = File(...),
    video_type: VideoType = Form(...),
    db: Session = Depends(get_db)
):
    """
    上传视频文件进行处理
    
    - **file**: 视频文件
    - **video_type**: 视频类型 (movie:电影, course:教学视频)
    """
    # 验证文件类型
    file_ext = os.path.splitext(file.filename)[1][1:].lower()
    if file_ext not in settings.SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件格式: {file_ext}。支持的格式: {', '.join(settings.SUPPORTED_FORMATS)}"
        )
    
    # 保存上传的文件
    file_path = os.path.join(UPLOAD_DIR, f"{file.filename}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 创建处理任务
    task = task_service.create_task(file_path, video_type, db)
    
    return VideoUploadResponse(
        task_id=task.id,
        status=task.status
    )

@router.post("/url", response_model=VideoUploadResponse, status_code=status.HTTP_201_CREATED)
async def process_video_url(request: VideoUrlRequest, db: Session = Depends(get_db)):
    """
    提交视频URL进行处理
    
    - **url**: 视频URL
    - **video_type**: 视频类型 (movie:电影, course:教学视频)
    """
    # 创建处理任务，视频将在后台下载
    task = task_service.create_task("", request.video_type, db, video_url=str(request.url))
    
    return VideoUploadResponse(
        task_id=task.id,
        status=task.status
    )

@router.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """
    获取任务处理状态
    
    - **task_id**: 任务ID
    """
    task = task_service.get_task(task_id, db)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到任务: {task_id}"
        )
    
    # 确保返回的对象包含所有必要的字段
    # 创建一个新的响应对象而不是直接返回task对象
    return TaskStatusResponse(
        id=task.id,
        status=task.status,
        progress=task.progress,
        message=task.message,
        output_path=task.result_path,
        created_at=str(task.created_at) if task.created_at else None,
        completed_at=str(task.updated_at) if task.updated_at else None,
        use_omni=False  # 数据库中可能没有这个字段，默认为False
    )

@router.get("/tasks/{task_id}/download")
async def download_result(task_id: str, db: Session = Depends(get_db)):
    """
    下载处理完成的视频
    
    - **task_id**: 任务ID
    """
    task = task_service.get_task(task_id, db)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到任务: {task_id}"
        )
    
    if task.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"任务未完成，当前状态: {task.status}"
        )
    
    if not task.result_path or not os.path.exists(task.result_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="处理结果文件不存在"
        )
    
    return FileResponse(
        path=task.result_path,
        filename=os.path.basename(task.result_path),
        media_type="video/mp4"
    )

@router.get("/tasks", response_model=List[TaskStatusResponse])
async def list_tasks(
    skip: int = 0, 
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    列出所有任务
    
    - **skip**: 跳过记录数
    - **limit**: 返回记录数
    """
    tasks = task_service.list_tasks(skip=skip, limit=limit, db=db)
    
    # 将每个Task转换为TaskStatusResponse对象
    response_tasks = []
    for task in tasks:
        response_tasks.append(TaskStatusResponse(
            id=task.id,
            status=task.status,
            progress=task.progress,
            message=task.message,
            output_path=task.result_path,
            created_at=str(task.created_at) if task.created_at else None,
            completed_at=str(task.updated_at) if task.updated_at else None,
            use_omni=False  # 数据库中可能没有这个字段，默认为False
        ))
    
    return response_tasks

@router.post("", response_model=TaskResponse)
def create_video_task(
    task_data: TaskCreate,
    db: Session = Depends(get_db)
):
    """创建新的视频处理任务"""
    try:
        # 生成唯一任务ID
        task_id = str(uuid.uuid4())
        
        # 创建新任务
        new_task = Task(
            id=task_id,
            video_path=task_data.video_path,
            video_url=task_data.video_url,
            video_type=task_data.video_type,
            status=TaskStatus.PENDING,
            use_omni=task_data.use_omni  # 添加是否使用全模态模型的标志
        )
        
        db.add(new_task)
        db.commit()
        
        # 启动后台处理任务
        threading.Thread(
            target=video_service.process_video, 
            args=(new_task, db)
        ).start()
        
        return {"task_id": task_id, "message": "任务已提交"}
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建任务失败: {str(e)}"
        ) 