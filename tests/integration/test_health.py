from unittest import mock
from fastapi.testclient import TestClient

from src.main import app

def test_health_check_endpoint():
    """Verify that GET /api/v1/health returns code 200 and the correct healthy status payload."""
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

def test_health_check_logging():
    """Verify health requests emit logs using the central logging system."""
    with mock.patch("src.api.v1.endpoints.health.logger.info") as mock_log_info:
        with TestClient(app) as client:
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            mock_log_info.assert_called_once_with("Received liveness/readiness verification request")
