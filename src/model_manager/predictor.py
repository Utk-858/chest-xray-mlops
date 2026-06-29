import torch

from src.core.config import ModelConfig
from src.model_manager.loader import load_pytorch_model, resolve_device

class ModelPredictor:
    """
    Wrapper executor class for performing thread-safe model inference.
    Loads the model state once on initialization and runs prediction forwards.
    """
    def __init__(self, model_config: ModelConfig, device_setting: str):
        self.model_config = model_config
        self.device = resolve_device(device_setting)
        self.model = load_pytorch_model(model_config, device_setting)

    def predict(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass on a preprocessed tensor.
        Handles device routing and batch dimensions.
        """
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        # Unsqueeze batch dimension [B, C, H, W] if 3D tensor [C, H, W] is passed
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)
        return outputs
