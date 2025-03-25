import sys
import os
import sqlite3
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入项目配置
from app.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def add_use_omni_column():
    """向Task表添加use_omni列"""
    try:
        # 解析数据库路径
        db_path = settings.DATABASE_URL.replace("sqlite:///", "")
        if not os.path.exists(db_path):
            logger.error(f"数据库文件不存在: {db_path}")
            return False
            
        logger.info(f"正在更新数据库: {db_path}")
        
        # 连接到数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
        if not cursor.fetchone():
            logger.error("tasks表不存在")
            conn.close()
            return False
        
        # 检查列是否已存在
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if "use_omni" in columns:
            logger.info("use_omni列已存在，无需添加")
            conn.close()
            return True
        
        # 添加use_omni列，默认为0 (False)
        logger.info("添加use_omni列...")
        cursor.execute("ALTER TABLE tasks ADD COLUMN use_omni BOOLEAN DEFAULT 0")
        conn.commit()
        
        # 确认添加成功
        cursor.execute("PRAGMA table_info(tasks)")
        columns = [col[1] for col in cursor.fetchall()]
        if "use_omni" in columns:
            logger.info("use_omni列添加成功")
            success = True
        else:
            logger.error("添加use_omni列失败")
            success = False
        
        conn.close()
        return success
        
    except Exception as e:
        logger.error(f"更新数据库失败: {str(e)}")
        return False

if __name__ == "__main__":
    if add_use_omni_column():
        logger.info("数据库更新成功")
        sys.exit(0)
    else:
        logger.error("数据库更新失败")
        sys.exit(1) 