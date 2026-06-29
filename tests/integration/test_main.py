from fastapi.testclient import TestClient
from src.main import app
from src.core.config import get_settings

def test_app_metadata():
    """Verify that the FastAPI application metadata matches configuration settings."""
    settings = get_settings()
    assert app.title == settings.app.name
    assert app.version == settings.app.version
    assert app.debug == settings.app.debug

def test_openapi_documentation_endpoints():
    """Verify that the Swagger UI and OpenAPI JSON document definitions are accessible."""
    with TestClient(app) as client:
        # Check docs page loading
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        assert "swagger" in docs_response.text.lower()
        
        # Check openapi json loading
        openapi_response = client.get("/openapi.json")
        assert openapi_response.status_code == 200
        assert openapi_response.json()["info"]["title"] == app.title
