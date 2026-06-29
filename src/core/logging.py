import logging
import logging.config
from contextvars import ContextVar
import os
import yaml

from src.core.config import get_settings

# Async-safe context variable to hold the request ID for the current request context
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    """
    Logging filter that dynamically injects the active request_id context variable
    into the log record attributes so it is parsed by formatters automatically.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get() or "N/A"
        return True

def setup_logging() -> None:
    """
    Initializes the centralized logging configuration system using python's dictConfig.
    Falls back gracefully to basicConfig if loading or parsing custom config fails.
    """
    settings = get_settings()
    config_path = os.path.join("config", "logging.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)

            # Override log levels using dynamic application settings
            log_level = settings.logging.level.upper()
            config_dict["loggers"][""]["level"] = log_level

            logging.config.dictConfig(config_dict)
            logging.getLogger(__name__).debug(
                f"Successfully initialized centralized logging at level '{log_level}'"
            )
            return
        except Exception as e:
            # Fall back to standard setup if YAML configs are corrupt
            logging.basicConfig(level=logging.INFO)
            logging.getLogger(__name__).warning(
                f"Failed to load logging config from {config_path}: {e}. Falling back to basicConfig."
            )
            return

    # Fall back if path is missing
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).warning(
        f"Logging config file not found at {config_path}. Falling back to basicConfig."
    )

def get_logger(name: str) -> logging.Logger:
    """
    Retrieves a logger instance by name.
    """
    return logging.getLogger(name)
