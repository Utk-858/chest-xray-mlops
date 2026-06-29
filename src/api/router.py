from fastapi import APIRouter

from src.api.v1.router import v1_router

# Base routing hub for the serving application.
api_router = APIRouter()

# Mount versioned API routes
api_router.include_router(v1_router, prefix="/v1")
