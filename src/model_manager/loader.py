import os
from pathlib import Path
import torch
import torchvision.models as tv_models

from src.core.config import ModelConfig
from src.core.logging import get_logger

logger = get_logger(__name__)

def resolve_device(device_setting: str) -> torch.device:
    """
    Resolves the execution hardware target.
    If 'auto' is provided, resolves to CUDA, MPS, or falls back to CPU.
    """
    if device_setting == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_setting
    return torch.device(device)

def load_pytorch_model(model_config: ModelConfig, device_setting: str) -> torch.nn.Module:
    """
    Instantiates a torchvision model architecture based on name and loads weights.
    Falls back gracefully to random initializations if weights are missing or corrupt.
    """
    device = resolve_device(device_setting)
    logger.info(f"Initializing model architecture '{model_config.name}' on device '{device}'...")

    # Architecture registry mapping
    model_name_lower = model_config.name.lower()
    if "resnet50" in model_name_lower:
        model = tv_models.resnet50(num_classes=model_config.num_classes)
    elif "densenet121" in model_name_lower:
        model = tv_models.densenet121(num_classes=model_config.num_classes)
    elif "resnet18" in model_name_lower:
        model = tv_models.resnet18(num_classes=model_config.num_classes)
    else:
        logger.warning(
            f"Unrecognized model architecture target '{model_config.name}'. "
            "Defaulting implementation template to ResNet18."
        )
        model = tv_models.resnet18(num_classes=2)

    # Load state dict
    weights_path = Path(model_config.path)
    if weights_path.exists():
        try:
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info(f"Successfully loaded model weights from '{weights_path}'")
        except Exception as e:
            logger.error(
                f"Failed to load weights state dict from '{weights_path}': {e}. "
                "Falling back to uninitialized random weights."
            )
    else:
        logger.warning(
            f"Model weights file not found at '{weights_path}'. "
            "Falling back to uninitialized random weights."
        )

    model.to(device)
    model.eval()
    return model
