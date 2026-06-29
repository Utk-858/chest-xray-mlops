import io
from PIL import Image
from fastapi.testclient import TestClient

from src.main import app

def test_predict_endpoint_success():
    """Verify that posting a valid chest X-ray image returns status code 200, schema payload, and request ID headers."""
    # Create valid mock image
    img = Image.new("RGB", (256, 256), color="red")
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    
    with TestClient(app) as client:
        # Test 1: Auto-generation of request ID in header response
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test_xray.png", img_byte_arr, "image/png")}
        )
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0
        
        json_data = response.json()
        assert "predicted_class" in json_data
        assert "confidence" in json_data
        assert "probabilities" in json_data
        assert json_data["predicted_class"] in ["Normal", "Opacity"]
        assert 0.0 <= json_data["confidence"] <= 1.0

        # Test 2: Reuse of incoming request ID header
        custom_id = "test-custom-uuid-string"
        img_byte_arr.seek(0)
        response_custom = client.post(
            "/api/v1/predict",
            headers={"X-Request-ID": custom_id},
            files={"file": ("test_xray.png", img_byte_arr, "image/png")}
        )
        assert response_custom.status_code == 200
        assert response_custom.headers.get("X-Request-ID") == custom_id

def test_predict_endpoint_invalid_file():
    """Verify that posting an invalid file format returns status code 400."""
    invalid_file = io.BytesIO(b"Corrupted or arbitrary text payload")
    
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict",
            files={"file": ("report.txt", invalid_file, "text/plain")}
        )
        assert response.status_code == 400
        assert "invalid image format" in response.json()["detail"].lower()
