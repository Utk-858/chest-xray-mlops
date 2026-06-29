import pytest
import torch

from src.core.config import ModelConfig
from src.model_manager.loader import resolve_device, load_pytorch_model
from src.model_manager.predictor import ModelPredictor
from src.model_manager.registry import ModelRegistry

def test_resolve_device():
    """Verify device resolution converts auto/specific flags to torch.device objects."""
    assert isinstance(resolve_device("cpu"), torch.device)
    assert resolve_device("cpu").type == "cpu"
    
    auto_device = resolve_device("auto")
    assert isinstance(auto_device, torch.device)
    assert auto_device.type in ["cpu", "cuda", "mps"]

def test_registry_resolution():
    """Verify registry correctly resolves active config parameters."""
    config = ModelRegistry.get_active_model_config()
    assert isinstance(config, ModelConfig)
    assert config.name == "chest_xray_resnet50"

def test_load_pytorch_model():
    """Verify load_pytorch_model instantiates a valid PyTorch module in eval mode."""
    config = ModelRegistry.get_active_model_config()
    model = load_pytorch_model(config, "cpu")
    assert isinstance(model, torch.nn.Module)
    assert not model.training  # Must be in eval mode (not model.training)

def test_model_predictor():
    """Verify ModelPredictor runs predictions and returns correctly shaped outputs."""
    config = ModelRegistry.get_active_model_config()
    predictor = ModelPredictor(config, "cpu")
    
    # Mock preprocessed tensor of size [Channels, Height, Width]
    input_tensor = torch.randn(3, 224, 224)
    
    output = predictor.predict(input_tensor)
    assert isinstance(output, torch.Tensor)
    # Output matches ResNet50 fc layer output mapping (batch size 1, num_classes 2)
    assert output.shape == (1, 2)
