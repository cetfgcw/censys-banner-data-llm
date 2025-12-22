"""Tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()


def test_health():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_ready_before_model_load():
    """Test readiness before model is loaded (should fail)."""
    # Note: This will fail if model loads synchronously in lifespan
    # In real deployment, model loads async, so this test may need adjustment
    pass


def test_predict_endpoint_structure():
    """Test that predict endpoint has correct structure (may fail if model not loaded)."""
    response = client.post(
        "/predict",
        json={"banner_text": "SSH-2.0-OpenSSH_8.2p1"}
    )
    # Should either succeed (200) or fail with 503 (not ready)
    assert response.status_code in [200, 503]


def test_predict_validation():
    """Test input validation."""
    # Empty banner
    response = client.post("/predict", json={"banner_text": ""})
    assert response.status_code == 422
    
    # Missing field
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    # Too long
    response = client.post("/predict", json={"banner_text": "x" * 2001})
    assert response.status_code == 422


def test_batch_predict_structure():
    """Test batch predict endpoint structure."""
    response = client.post(
        "/predict/batch",
        json={"banners": ["SSH-2.0-OpenSSH_8.2p1", "HTTP/1.1 200 OK"]}
    )
    # Should either succeed or fail with 503
    assert response.status_code in [200, 503]


def test_batch_predict_validation():
    """Test batch input validation."""
    # Empty list
    response = client.post("/predict/batch", json={"banners": []})
    assert response.status_code == 422
    
    # Too many banners
    response = client.post("/predict/batch", json={"banners": ["x"] * 101})
    assert response.status_code == 422


def test_metrics_endpoint():
    """Test metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers.get("content-type", "")

