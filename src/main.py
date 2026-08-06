from contextlib import asynccontextmanager
import time
import uuid
from fastapi import FastAPI, Request, Response, HTTPException, status
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.api.router import api_router
from src.core.config import get_settings
from src.core.logging import get_logger, setup_logging, request_id_var
from src.model_manager.predictor import ModelPredictor
from src.metrics import (
    record_prediction_request,
    record_prediction_success,
    record_prediction_failure,
    observe_request_latency,
)

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

# HTTP middleware to record Prometheus metrics for prediction endpoints
@app.middleware("http")
async def record_api_metrics(request: Request, call_next):
    settings = get_settings()
    if not settings.metrics.enabled:
        return await call_next(request)

    path = request.url.path
    if path not in ["/api/v1/predict", "/api/v1/predict/batch"]:
        return await call_next(request)

    model_name = settings.model.name
    model_version = settings.model.version

    # Record total prediction request count
    record_prediction_request(model_name, model_version)

    start_time = time.perf_counter()
    try:
        response = await call_next(request)
        latency = time.perf_counter() - start_time
        observe_request_latency(model_name, model_version, latency)

        if response.status_code >= 400:
            record_prediction_failure(model_name, model_version)
        else:
            record_prediction_success(model_name, model_version)
        return response
    except Exception as e:
        latency = time.perf_counter() - start_time
        observe_request_latency(model_name, model_version, latency)
        record_prediction_failure(model_name, model_version)
        raise e

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

# Prometheus metrics scraping endpoint
@app.get("/metrics", summary="Retrieve Prometheus application telemetry metrics")
async def metrics():
    settings = get_settings()
    if not settings.metrics.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics collection is currently disabled in application settings."
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Register API Router
app.include_router(api_router, prefix="/api")
