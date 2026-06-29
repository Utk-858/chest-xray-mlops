import os
from unittest import mock
import pytest
from pydantic import ValidationError
from src.core.config import get_settings, Settings

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Automatically clear settings cache before and after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()

def test_default_config_loading():
    """Verify that settings correctly load default values from config/config.yaml."""
    settings = get_settings()
    assert settings.app.name == "chest-xray-mlops"
    assert settings.app.version == "1.0.0"
    assert settings.app.debug is False
    assert settings.inference.device == "auto"
    assert settings.inference.confidence_threshold == 0.5
    assert settings.model.name == "chest_xray_resnet50"
    assert settings.model.input_size == (224, 224)

def test_env_variable_overrides():
    """Verify environment variables take precedence over config.yaml via nested delimiters."""
    env_vars = {
        "APP__DEBUG": "True",
        "APP__PORT": "9000",
        "INFERENCE__DEVICE": "cuda",
        "INFERENCE__CONFIDENCE_THRESHOLD": "0.75",
        "ACTIVE_MODEL": "resnet50",
    }
    with mock.patch.dict(os.environ, env_vars):
        settings = get_settings()
        assert settings.app.debug is True
        assert settings.app.port == 9000
        assert settings.inference.device == "cuda"
        assert settings.inference.confidence_threshold == 0.75

def test_active_model_validation_error():
    """Verify that selecting an unconfigured active model throws a ValidationError at startup (fail-fast)."""
    env_vars = {
        "ACTIVE_MODEL": "non_existent_model",
    }
    with mock.patch.dict(os.environ, env_vars):
        with pytest.raises(ValidationError) as exc_info:
            get_settings()
        assert "is not configured" in str(exc_info.value)

def test_validation_errors():
    """Verify field-level type validation and value range checks throw validation errors."""
    # Port out of bounds
    with mock.patch.dict(os.environ, {"APP__PORT": "99999"}):
        with pytest.raises(ValidationError):
            Settings()

    # Confidence threshold out of bounds
    with mock.patch.dict(os.environ, {"INFERENCE__CONFIDENCE_THRESHOLD": "1.5"}):
        with pytest.raises(ValidationError):
            Settings()

    # Unsupported hardware device
    with mock.patch.dict(os.environ, {"INFERENCE__DEVICE": "invalid_device"}):
        with pytest.raises(ValidationError):
            Settings()
