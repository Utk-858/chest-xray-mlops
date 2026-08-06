import io
import os
from unittest import mock
from PIL import Image
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from src.main import app
from src.core.config import get_settings

# Verify metrics implementation integration

def test_metrics_endpoint_basic():
    """Verify that the /metrics endpoint returns HTTP 200 and Prometheus text format."""
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        assert "chest_xray_prediction_requests_total" in response.text

def test_metrics_increment_on_prediction():
    """Verify that a successful prediction request increments the request and success counters."""
    # Create valid mock image
    img = Image.new("RGB", (224, 224), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    settings = get_settings()
    labels = {
        "model_name": settings.model.name,
        "model_version": settings.model.version
    }

    # Capture initial values
    init_requests = REGISTRY.get_sample_value("chest_xray_prediction_requests_total", labels) or 0.0
    init_success = REGISTRY.get_sample_value("chest_xray_prediction_success_total", labels) or 0.0
    init_failures = REGISTRY.get_sample_value("chest_xray_prediction_failures_total", labels) or 0.0

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test_xray.png", buf, "image/png")}
        )
        assert response.status_code == 200

        # Capture final values
        new_requests = REGISTRY.get_sample_value("chest_xray_prediction_requests_total", labels) or 0.0
        new_success = REGISTRY.get_sample_value("chest_xray_prediction_success_total", labels) or 0.0
        new_failures = REGISTRY.get_sample_value("chest_xray_prediction_failures_total", labels) or 0.0

        assert new_requests == init_requests + 1
        assert new_success == init_success + 1
        assert new_failures == init_failures  # no failure increment

        # Verify pipeline latency samples were registered
        count_val = REGISTRY.get_sample_value("chest_xray_preprocessing_seconds_count", labels) or 0.0
        assert count_val > 0.0

def test_metrics_increment_on_failure():
    """Verify that a failed prediction request increments the request and failure counters."""
    # Invalid file payload
    invalid_file = io.BytesIO(b"Not an image")

    settings = get_settings()
    labels = {
        "model_name": settings.model.name,
        "model_version": settings.model.version
    }

    # Capture initial values
    init_requests = REGISTRY.get_sample_value("chest_xray_prediction_requests_total", labels) or 0.0
    init_success = REGISTRY.get_sample_value("chest_xray_prediction_success_total", labels) or 0.0
    init_failures = REGISTRY.get_sample_value("chest_xray_prediction_failures_total", labels) or 0.0

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("report.txt", invalid_file, "text/plain")}
        )
        assert response.status_code == 400

        # Capture final values
        new_requests = REGISTRY.get_sample_value("chest_xray_prediction_requests_total", labels) or 0.0
        new_success = REGISTRY.get_sample_value("chest_xray_prediction_success_total", labels) or 0.0
        new_failures = REGISTRY.get_sample_value("chest_xray_prediction_failures_total", labels) or 0.0

        assert new_requests == init_requests + 1
        assert new_success == init_success  # no success increment
        assert new_failures == init_failures + 1

def test_metrics_disabled_endpoint():
    """Verify that the /metrics endpoint returns HTTP 404 when disabled."""
    env_vars = {"METRICS__ENABLED": "False"}
    with mock.patch.dict(os.environ, env_vars):
        get_settings.cache_clear()
        try:
            with TestClient(app) as client:
                response = client.get("/metrics")
                assert response.status_code == 404
        finally:
            get_settings.cache_clear()
