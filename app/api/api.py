from fastapi import APIRouter

from app.api.endpoints import videos

api_router = APIRouter()
api_router.include_router(videos.router, prefix="/videos", tags=["videos"]) 