import os
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.api.api import api_router
from app.config import settings, OUTPUT_DIR
from app.core.database import init_db

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# 创建FastAPI应用
app = FastAPI(
    title=settings.APP_NAME,
    description="自动视频剪辑与解说API",
    version="0.1.0",
)

# 允许CORS (跨域资源共享)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(api_router, prefix=settings.API_V1_STR)

# 挂载静态文件目录（用于结果视频访问）
app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"全局异常: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": f"服务器内部错误: {str(exc)}"}
    )

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    # 确保目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 初始化数据库
    init_db()
    
    logging.info("应用启动成功")

# 首页路由
@app.get("/")
async def read_root():
    return {
        "message": "欢迎使用视频自动剪辑解说API",
        "docs_url": "/docs",
        "api_version": "v1"
    } 