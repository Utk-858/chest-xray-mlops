import os
from functools import lru_cache
from typing import Type
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

class AppConfig(BaseModel):
    name: str = "chest-xray-mlops"
    version: str = "1.0.0"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    debug: bool = False

class ModelConfig(BaseModel):
    name: str
    version: str
    path: str
    input_size: tuple[int, int] = (224, 224)
    num_classes: int

class InferenceConfig(BaseModel):
    device: str = Field(default="auto", pattern="^(cpu|cuda|mps|auto)$")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    batch_size: int = Field(default=1, ge=1)

class LoggingConfig(BaseModel):
    level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

class Settings(BaseSettings):
    app: AppConfig = AppConfig()
    active_model: str
    models: dict[str, ModelConfig] = Field(default_factory=dict)
    inference: InferenceConfig = InferenceConfig()
    logging: LoggingConfig = LoggingConfig()

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="",
    )

    @model_validator(mode="after")
    def validate_active_model_exists(self) -> "Settings":
        """Verify that the selected active model has a matching configuration."""
        if self.active_model not in self.models:
            raise ValueError(
                f"active_model '{self.active_model}' is not configured. "
                f"Available models: {list(self.models.keys())}"
            )
        return self

    @property
    def model(self) -> ModelConfig:
        """Helper property to access the configuration of the currently active model."""
        # Guaranteed to exist because of model validator at startup
        return self.models[self.active_model]

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        yaml_file = os.getenv("CONFIG_PATH", "config/config.yaml")
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
        )

@lru_cache()
def get_settings() -> Settings:
    """Exposes settings instance with caching."""
    return Settings()
