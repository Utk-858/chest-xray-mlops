import io
from PIL import Image
from fastapi.testclient import TestClient
import pytest

from src.main import app

def test_batch_predict_success():
    """Verify that posting multiple valid images returns a 200 OK and valid results for all items."""
    img1 = Image.new("RGB", (224, 224), color="white")
    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")
    buf1.seek(0)

    img2 = Image.new("RGB", (224, 224), color="black")
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    buf2.seek(0)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/batch",
            files=[
                ("files", ("img1.png", buf1, "image/png")),
                ("files", ("img2.png", buf2, "image/png"))
            ]
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "img1.png" in data["results"]
        assert "img2.png" in data["results"]

        # Assert results match PredictionResult schema structure
        for filename in ["img1.png", "img2.png"]:
            res = data["results"][filename]
            assert "predicted_class" in res
            assert "confidence" in res
            assert "probabilities" in res
            assert res["predicted_class"] in ["Normal", "Opacity"]

def test_batch_predict_partial_failures():
    """Verify that a batch request with one valid image and one corrupt file succeeds with partial outputs."""
    # 1. Valid image
    img = Image.new("RGB", (224, 224), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    # 2. Corrupt/Invalid txt file
    corrupt = io.BytesIO(b"Not an image file")

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predict/batch",
            files=[
                ("files", ("valid.png", buf, "image/png")),
                ("files", ("corrupt.txt", corrupt, "text/plain"))
            ]
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "valid.png" in data["results"]
        assert "corrupt.txt" in data["results"]

        # Valid image yields structured prediction
        valid_res = data["results"]["valid.png"]
        assert "predicted_class" in valid_res
        assert valid_res["predicted_class"] in ["Normal", "Opacity"]

        # Invalid file yields descriptive error string
        corrupt_res = data["results"]["corrupt.txt"]
        assert isinstance(corrupt_res, str)
        assert "invalid image format" in corrupt_res.lower()
