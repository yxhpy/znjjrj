import uuid
import threading
import logging
from datetime import datetime
from sqlalchemy.orm import Session

from app.core.database import Task, TaskStatus, VideoType
from app.services.video_service import video_service

logger = logging.getLogger(__name__)

class TaskService:
    """任务管理服务"""
    
    def create_task(self, video_path: str, video_type: VideoType, db: Session, video_url: str = None) -> Task:
        """创建新的视频处理任务"""
        task_id = str(uuid.uuid4())
        
        # 创建任务记录
        task = Task(
            id=task_id,
            video_path=video_path,
            video_url=video_url,
            video_type=video_type,
            status=TaskStatus.PENDING,
            progress=0.0,
            message="任务已创建，等待处理",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # 保存到数据库
        db.add(task)
        db.commit()
        db.refresh(task)
        
        # 在新线程中启动视频处理任务
        threading.Thread(target=self._process_video_task, args=(task_id, db)).start()
        
        return task
    
    def _process_video_task(self, task_id: str, db: Session) -> None:
        """在后台处理视频任务"""
        # 创建新的数据库会话
        local_db = Session(bind=db.get_bind())
        
        try:
            # 获取任务
            task = local_db.query(Task).filter(Task.id == task_id).first()
            if not task:
                logger.error(f"找不到任务: {task_id}")
                return
            
            # 处理视频
            video_service.process_video(task, local_db)
            
        except Exception as e:
            logger.error(f"任务处理失败: {str(e)}")
            
            # 更新任务状态为失败
            task = local_db.query(Task).filter(Task.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED
                task.message = f"处理失败: {str(e)}"
                local_db.commit()
        finally:
            local_db.close()
    
    def get_task(self, task_id: str, db: Session) -> Task:
        """获取任务状态"""
        return db.query(Task).filter(Task.id == task_id).first()
    
    def list_tasks(self, skip: int = 0, limit: int = 100, db: Session = None) -> list:
        """列出所有任务"""
        return db.query(Task).offset(skip).limit(limit).all()

# 创建任务服务单例
task_service = TaskService() 