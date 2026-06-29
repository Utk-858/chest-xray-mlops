from fastapi import APIRouter

from src.core.logging import get_logger
from src.schemas.health import HealthResponse

logger = get_logger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse, summary="Perform liveness/readiness assertion check")
async def get_health() -> HealthResponse:
    """
    Returns liveness/readiness indicators of the ML inference serving application.
    """
    logger.info("Received liveness/readiness verification request")
    return HealthResponse(status="healthy")
