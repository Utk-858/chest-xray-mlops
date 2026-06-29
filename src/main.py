from contextlib import asynccontextmanager
import uuid
from fastapi import FastAPI, Request

from src.api.router import api_router
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging, request_id_var
from src.model_manager.predictor import ModelPredictor

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the startup and shutdown lifecycle events of the FastAPI application.
    """
    # Initialize the logging system
    setup_logging()
    
    logger.info("Initializing Chest X-Ray inference serving application bootstrap...")
    settings = get_settings()
    logger.info(
        f"Application loaded: {settings.app.name} "
        f"[Env: {settings.app.environment}, Debug: {settings.app.debug}]"
    )
    
    # Initialize and cache model predictor in application state
    logger.info("Instantiating active model predictor...")
    app.state.predictor = ModelPredictor(
        model_config=settings.model,
        device_setting=settings.inference.device,
    )
    
    yield
    
    logger.info("Shutting down Chest X-Ray inference serving application...")

settings = get_settings()

app = FastAPI(
    title=settings.app.name,
    version=settings.app.version,
    debug=settings.app.debug,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# HTTP middleware to assign unique request IDs to contextvars and HTTP headers
@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    # Retrieve request ID from incoming request headers or generate a new UUID
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)

# Register API Router
app.include_router(api_router, prefix="/api")
