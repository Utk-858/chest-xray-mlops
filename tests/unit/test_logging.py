import logging
from unittest import mock
from src.core.logging import setup_logging, get_logger
from src.core.config import get_settings

def test_get_logger():
    """Verify that get_logger returns a valid standard python logger with correct namespacing."""
    logger = get_logger("test_namespace")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_namespace"

def test_setup_logging_success():
    """Verify that setup_logging configures root log levels dynamically from settings."""
    settings = get_settings()
    
    with mock.patch("logging.config.dictConfig") as mock_dict_config:
        setup_logging()
        mock_dict_config.assert_called_once()
        
        # Verify dictConfig gets called with overridden settings values
        config = mock_dict_config.call_args[0][0]
        assert config["loggers"][""]["level"] == settings.logging.level

def test_setup_logging_fallback():
    """Verify that setup_logging falls back to standard basicConfig if logging.yaml is missing."""
    with mock.patch("os.path.exists", return_value=False):
        with mock.patch("logging.basicConfig") as mock_basic_config:
            setup_logging()
            mock_basic_config.assert_called_once_with(level=logging.INFO)
