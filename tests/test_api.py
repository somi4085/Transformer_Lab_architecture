
import pytest

from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():

    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_generate_endpoint(client):

    response = client.post(
        "/generate",
        json={
            "text": "Once upon a time",
            "max_new_tokens": 10
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["input_text"] == "Once upon a time"
    assert isinstance(data["generated_text"], str)
    assert data["generated_token_count"] <= 10


def test_empty_input(client):

    response = client.post(
        "/generate",
        json={
            "text": "",
            "max_new_tokens": 10
        }
    )

    assert response.status_code == 422


def test_invalid_token_limit(client):

    response = client.post(
        "/generate",
        json={
            "text": "Hello",
            "max_new_tokens": 100
        }
    )

    assert response.status_code == 422


def test_missing_text(client):

    response = client.post(
        "/generate",
        json={
            "max_new_tokens": 10
        }
    )

    assert response.status_code == 422


def test_multiple_requests_same_model(client):

    service_before = app.state.inference_service

    for _ in range(3):

        response = client.post(
            "/generate",
            json={
                "text": "A little dog",
                "max_new_tokens": 5
            }
        )

        assert response.status_code == 200

    service_after = app.state.inference_service

    assert service_before is service_after


if __name__ == "__main__":

    pytest.main([__file__, "-v"])
