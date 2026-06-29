from fastapi import APIRouter

from src.api.v1.endpoints import health, predict

v1_router = APIRouter()

# Include versioned API endpoint routers
v1_router.include_router(health.router, tags=["health"])
v1_router.include_router(predict.router, tags=["predict"])
