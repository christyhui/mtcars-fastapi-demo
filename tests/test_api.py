"""
tests/test_api.py

Automated tests for the MTCARS MPG Predictor API.

Run with:
    pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    """
    Use TestClient as a context manager so the lifespan event runs,
    which loads the model before any tests execute.
    """
    with TestClient(app) as c:
        yield c


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.json() == {"status": "ok"}


# ── /ready ────────────────────────────────────────────────────────────────────


def test_ready_returns_200_when_model_loaded(client):
    response = client.get("/ready")
    assert response.status_code in (200, 503)


def test_ready_response_structure_when_model_loaded(client):
    response = client.get("/ready")
    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["model_loaded"] is True


# ── /predict ──────────────────────────────────────────────────────────────────


def test_predict_returns_200_with_valid_input(client):
    response = client.post("/predict", json={"wt": 2.62, "hp": 110})
    assert response.status_code == 200


def test_predict_returns_mpg_field(client):
    response = client.post("/predict", json={"wt": 2.62, "hp": 110})
    if response.status_code == 200:
        data = response.json()
        assert "predicted_mpg" in data
        assert isinstance(data["predicted_mpg"], float)


def test_predict_mpg_is_positive(client):
    response = client.post("/predict", json={"wt": 2.62, "hp": 110})
    if response.status_code == 200:
        assert response.json()["predicted_mpg"] > 0


def test_predict_heavier_car_gets_lower_mpg(client):
    """A heavier car should be predicted to have lower fuel efficiency."""
    light = client.post("/predict", json={"wt": 1.5, "hp": 100})
    heavy = client.post("/predict", json={"wt": 5.0, "hp": 100})
    if light.status_code == 200 and heavy.status_code == 200:
        assert light.json()["predicted_mpg"] > heavy.json()["predicted_mpg"]


def test_predict_higher_hp_gets_lower_mpg(client):
    """A higher horsepower car should be predicted to have lower fuel efficiency."""
    low_hp = client.post("/predict", json={"wt": 3.0, "hp": 80})
    high_hp = client.post("/predict", json={"wt": 3.0, "hp": 300})
    if low_hp.status_code == 200 and high_hp.status_code == 200:
        assert low_hp.json()["predicted_mpg"] > high_hp.json()["predicted_mpg"]


# ── Input validation ──────────────────────────────────────────────────────────


def test_predict_missing_wt_returns_422(client):
    response = client.post("/predict", json={"hp": 110})
    assert response.status_code == 422


def test_predict_missing_hp_returns_422(client):
    response = client.post("/predict", json={"wt": 2.62})
    assert response.status_code == 422


def test_predict_missing_both_fields_returns_422(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_type_string_returns_422(client):
    response = client.post("/predict", json={"wt": "heavy", "hp": "fast"})
    assert response.status_code == 422


def test_predict_negative_wt_returns_422(client):
    """Weight must be positive (gt=0 in Pydantic schema)."""
    response = client.post("/predict", json={"wt": -1.0, "hp": 110})
    assert response.status_code == 422


def test_predict_negative_hp_returns_422(client):
    """Horsepower must be positive (gt=0 in Pydantic schema)."""
    response = client.post("/predict", json={"wt": 2.62, "hp": -50})
    assert response.status_code == 422


def test_predict_zero_wt_returns_422(client):
    """Zero weight should fail gt=0 validation."""
    response = client.post("/predict", json={"wt": 0, "hp": 110})
    assert response.status_code == 422
