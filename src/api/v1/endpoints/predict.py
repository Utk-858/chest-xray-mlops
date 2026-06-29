import io
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, status
from PIL import Image

from src.api.dependencies import get_predictor
from src.core.logging import get_logger
from src.model_manager.predictor import ModelPredictor
from src.schemas.predict import PredictionResult, BatchPredictionResult
from src.services.inference import InferenceService

logger = get_logger(__name__)
router = APIRouter()

@router.post(
    "/predict",
    response_model=PredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Predict pathology from Chest X-Ray image",
)
async def predict_chest_xray(
    file: UploadFile = File(..., description="Uploaded raw chest X-ray image"),
    predictor: ModelPredictor = Depends(get_predictor),
) -> PredictionResult:
    """
    Accepts an uploaded image file, validates its format, executes model inference via the orchestration service,
    and returns a structured prediction payload.
    """
    logger.info(f"Received prediction request for file: {file.filename}")

    # 1. Read file bytes
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload file contents: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file contents."
        )

    # 2. Validate image format by attempting PIL open
    try:
        image = Image.open(io.BytesIO(contents))
        image.verify()  # Verify that the image is valid
        # Re-open after verify() because verify() closes the file descriptor and invalidates the image state
        image = Image.open(io.BytesIO(contents))
    except Exception as e:
        logger.warning(f"Uploaded file failed image verification: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format. Please upload a valid PNG, JPG, or JPEG file."
        )

    # 3. Coordinate Inference Service
    try:
        inference_service = InferenceService(predictor)
        result = inference_service.predict(image)
        logger.info(
            f"Successfully processed prediction: {result.predicted_class} "
            f"(confidence: {result.confidence:.4f})"
        )
        return result
    except Exception as e:
        logger.error(f"Inference pipeline execution error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error occurred while executing inference pipeline."
        )

@router.post(
    "/predict/batch",
    response_model=BatchPredictionResult,
    status_code=status.HTTP_200_OK,
    summary="Batch predict pathology from multiple uploaded Chest X-Ray images",
)
async def predict_chest_xray_batch(
    files: list[UploadFile] = File(..., description="List of raw chest X-ray images"),
    predictor: ModelPredictor = Depends(get_predictor),
) -> BatchPredictionResult:
    """
    Accepts multiple uploaded image files, processes valid ones in configured batches,
    captures errors for invalid items, and returns mapped results.
    """
    logger.info(f"Received batch prediction request containing {len(files)} files")

    results: dict[str, PredictionResult | str] = {}
    valid_images: list[tuple[str, Image.Image]] = []

    for file in files:
        # Read file bytes
        try:
            contents = await file.read()
        except Exception as e:
            logger.warning(f"Failed to read batch upload item '{file.filename}': {e}")
            results[file.filename] = f"Failed to read file contents: {e}"
            continue

        # Open image and verify format
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
            image = Image.open(io.BytesIO(contents))
            valid_images.append((file.filename, image))
        except Exception as e:
            logger.warning(f"Batch item '{file.filename}' failed image validation checks: {e}")
            results[file.filename] = f"Invalid image format: {e}"

    # Execute batch prediction
    if valid_images:
        try:
            inference_service = InferenceService(predictor)
            batch_results = inference_service.predict_batch(valid_images)
            results.update(batch_results)
        except Exception as e:
            logger.error(f"Batch pipeline runtime error: {e}", exc_info=True)
            # Mark all valid images as failed due to serving issue
            for name, _ in valid_images:
                results[name] = f"Pipeline execution failed: {e}"

    logger.info(f"Successfully processed batch prediction request for {len(files)} files")
    return BatchPredictionResult(results=results)
