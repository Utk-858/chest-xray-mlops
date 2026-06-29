from src.core.config import get_settings, ModelConfig

class ModelRegistry:
    """
    Manages resolution of model configurations from active application settings.
    """
    @staticmethod
    def get_active_model_config() -> ModelConfig:
        """
        Retrieves the configuration properties of the currently active model.
        """
        settings = get_settings()
        return settings.model

    @staticmethod
    def get_model_config(model_id: str) -> ModelConfig:
        """
        Retrieves configuration properties for a specific model ID.
        """
        settings = get_settings()
        if model_id not in settings.models:
            raise ValueError(f"Model ID '{model_id}' is not configured in settings.")
        return settings.models[model_id]
