from unittest import mock
from PIL import Image
import pytest
import torch

from src.pipeline.preprocessing import preprocess_image
from src.pipeline.postprocessing import postprocess_outputs
from src.services.inference import InferenceService
from src.schemas.predict import PredictionResult

def test_preprocessing():
    """Verify preprocessing converts different PIL image modes to standardized tensors."""
    # Test RGB image normalization
    rgb_img = Image.new("RGB", (400, 300), color="white")
    tensor = preprocess_image(rgb_img, target_size=(224, 224))
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)

    # Test Grayscale image conversion (must yield 3 channels for ResNet)
    gray_img = Image.new("L", (150, 150), color=128)
    tensor_gray = preprocess_image(gray_img, target_size=(224, 224))
    assert tensor_gray.shape == (3, 224, 224)

def test_postprocessing():
    """Verify postprocessing resolves logit tensors to Pydantic PredictionResult structures."""
    # Normal (index 0) is dominant
    logits_normal = torch.tensor([[5.0, 1.0]])
    result = postprocess_outputs(logits_normal, confidence_threshold=0.5)
    assert isinstance(result, PredictionResult)
    assert result.predicted_class == "Normal"
    assert result.confidence > 0.95
    assert pytest.approx(sum(result.probabilities.values())) == 1.0

    # Opacity (index 1) is dominant
    logits_opacity = torch.tensor([[0.5, 4.5]])
    result_opacity = postprocess_outputs(logits_opacity, confidence_threshold=0.5)
    assert result_opacity.predicted_class == "Opacity"
    assert result_opacity.confidence > 0.95

def test_inference_service_orchestration():
    """Verify InferenceService successfully coordinates preprocessing, prediction, and postprocessing."""
    # Create mock predictor
    mock_predictor = mock.MagicMock()
    mock_predictor.predict.return_value = torch.tensor([[10.0, -10.0]])

    with mock.patch("src.services.inference.logger.info") as mock_logger_info:
        service = InferenceService(mock_predictor)
        test_img = Image.new("RGB", (256, 256), color="red")

        result = service.predict(test_img)

        # Asserts
        assert isinstance(result, PredictionResult)
        assert result.predicted_class == "Normal"
        assert result.confidence == pytest.approx(1.0)
        mock_predictor.predict.assert_called_once()

        # Assert logging emitted expected latency stats format
        mock_logger_info.assert_called_once()
        log_msg = mock_logger_info.call_args[0][0]
        assert "Prediction Performance Log - " in log_msg
        assert "RequestID:" in log_msg
        assert "Model:" in log_msg
        assert "Device:" in log_msg
        assert "Prediction:" in log_msg
        assert "Latency Breakdown - Total:" in log_msg
        assert "Preprocessing:" in log_msg
        assert "Model Inference:" in log_msg
        assert "Postprocessing:" in log_msg


