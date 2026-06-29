import time
from PIL import Image
import torch

from src.core.config import get_settings
from src.core.logging import get_logger, request_id_var
from src.model_manager.predictor import ModelPredictor
from src.pipeline.preprocessing import preprocess_image
from src.pipeline.postprocessing import postprocess_outputs
from src.schemas.predict import PredictionResult
from src.utils.timing import time_block

logger = get_logger(__name__)

class InferenceService:
    """
    Orchestration layer that maps raw PIL image inputs to validated PredictionResult schemas.
    Maintains isolation between FastAPI endpoints and pipeline implementations.
    Supports both single-image and chunked batch inference requests.
    """
    def __init__(self, predictor: ModelPredictor):
        self.predictor = predictor
        self.settings = get_settings()

    def predict(self, image: Image.Image) -> PredictionResult:
        """
        Coordinates the execution steps of the inference pipeline:
        1. Runs transformations on raw PIL images.
        2. Feeds preprocessed tensors into the model predictor.
        3. Parses model output logits to structured class probabilities.
        Instruments code blocks to log detailed latency and execution context.
        """
        total_start = time.perf_counter()

        # Step 1: Preprocessing
        with time_block() as preprocess_stats:
            target_size = self.settings.model.input_size
            preprocessed_tensor = preprocess_image(image, target_size=target_size)

        # Step 2: Prediction
        with time_block() as predict_stats:
            logits = self.predictor.predict(preprocessed_tensor)

        # Step 3: Postprocessing
        with time_block() as postprocess_stats:
            confidence_threshold = self.settings.inference.confidence_threshold
            result = postprocess_outputs(logits, confidence_threshold=confidence_threshold)

        total_end = time.perf_counter()
        total_time_ms = (total_end - total_start) * 1000.0

        # Log detailed request context, metadata, and performance breakdown
        logger.info(
            f"Prediction Performance Log - "
            f"RequestID: {request_id_var.get() or 'N/A'} | "
            f"Model: {self.settings.model.name} (v{self.settings.model.version}) | "
            f"Device: {self.predictor.device} | "
            f"Prediction: {result.predicted_class} (confidence: {result.confidence:.4f}) | "
            f"Latency Breakdown - Total: {total_time_ms:.2f}ms | "
            f"Preprocessing: {preprocess_stats['elapsed_ms']:.2f}ms | "
            f"Model Inference: {predict_stats['elapsed_ms']:.2f}ms | "
            f"Postprocessing: {postprocess_stats['elapsed_ms']:.2f}ms"
        )

        return result

    def predict_batch(self, images: list[tuple[str, Image.Image]]) -> dict[str, PredictionResult | str]:
        """
        Orchestrates batch inference over a list of named PIL images.
        Stacks valid tensors to perform parallel forward passes under configured limits,
        handling item preprocessing or evaluation failures gracefully.
        """
        results: dict[str, PredictionResult | str] = {}
        valid_items: list[tuple[str, torch.Tensor]] = []
        
        total_start = time.perf_counter()
        preprocess_latencies = []

        # Step 1: Preprocess items sequentially, capturing failures
        for name, img in images:
            try:
                with time_block() as p_stats:
                    target_size = self.settings.model.input_size
                    tensor = preprocess_image(img, target_size=target_size)
                valid_items.append((name, tensor))
                preprocess_latencies.append(p_stats["elapsed_ms"])
            except Exception as e:
                logger.warning(f"Batch item '{name}' failed preprocessing validation: {e}")
                results[name] = f"Preprocessing failed: {e}"

        if not valid_items:
            logger.info("Batch inference completed with 0 successfully preprocessed items.")
            return results

        # Step 2: Slice valid items into chunks according to configured batch size
        config_batch_size = self.settings.inference.batch_size
        batches = [
            valid_items[i:i + config_batch_size] 
            for i in range(0, len(valid_items), config_batch_size)
        ]

        predict_latencies = []
        postprocess_latencies = []

        # Step 3: Process batches
        for batch in batches:
            names = [item[0] for item in batch]
            tensors = [item[1] for item in batch]

            # Stack tensors to leverage hardware vectorization (Shape: [B, C, H, W])
            batch_tensor = torch.stack(tensors)

            # Predict batch
            try:
                with time_block() as pred_stats:
                    batch_logits = self.predictor.predict(batch_tensor)
                predict_latencies.append(pred_stats["elapsed_ms"])
            except Exception as e:
                logger.error(f"Batch prediction forward pass failed: {e}")
                for name in names:
                    results[name] = f"Model forward pass failed: {e}"
                continue

            # Postprocess individual logits sequentially
            confidence_threshold = self.settings.inference.confidence_threshold
            for idx, name in enumerate(names):
                try:
                    with time_block() as post_stats:
                        logit = batch_logits[idx].unsqueeze(0)
                        res = postprocess_outputs(logit, confidence_threshold=confidence_threshold)
                    results[name] = res
                    postprocess_latencies.append(post_stats["elapsed_ms"])
                except Exception as e:
                    logger.error(f"Postprocessing failed for item '{name}': {e}")
                    results[name] = f"Postprocessing failed: {e}"

        total_end = time.perf_counter()
        total_time_ms = (total_end - total_start) * 1000.0

        # Log timing metrics summaries
        avg_preprocess = sum(preprocess_latencies) / len(preprocess_latencies) if preprocess_latencies else 0.0
        avg_predict = sum(predict_latencies) / len(predict_latencies) if predict_latencies else 0.0
        avg_postprocess = sum(postprocess_latencies) / len(postprocess_latencies) if postprocess_latencies else 0.0

        logger.info(
            f"Batch Latency Breakdown - Total Request: {total_time_ms:.2f}ms | "
            f"Processed Items: {len(valid_items)}/{len(images)} | "
            f"Avg Preprocessing: {avg_preprocess:.2f}ms/item | "
            f"Avg Model Inference: {avg_predict:.2f}ms/batch | "
            f"Avg Postprocessing: {avg_postprocess:.2f}ms/item"
        )

        return results
