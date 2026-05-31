from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.api_inference import app

client = TestClient(app)


def test_running_endpoint():
    response = client.get("/running")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_endpoint():
    mock_agent = MagicMock()
    mock_agent.generate_sql.return_value = "SELECT * FROM users"

    payload = {
        "question": "show all users",
        "connection_string": "sqlite:///test.db",
    }

    with patch(
        "src.api.api_inference.get_agent",
        return_value=mock_agent,
    ):
        response = client.post("/generate", json=payload)

    assert response.status_code == 200
    assert response.json()["sql"] == "SELECT * FROM users"


def test_query_endpoint():
    mock_agent = MagicMock()
    mock_agent.query.return_value = {
        "sql": "SELECT * FROM users",
        "rows": [{"id": 1}],
    }

    payload = {
        "question": "show all users",
        "connection_string": "sqlite:///test.db",
    }

    with patch(
        "src.api.api_inference.get_agent",
        return_value=mock_agent,
    ):
        response = client.post("/query", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "sql" in data
    assert "rows" in data


def test_generate_endpoint_model_loading_failed():
    payload = {
        "question": "show users",
        "connection_string": "sqlite:///test.db",
    }

    with patch(
        "src.api.api_inference.get_agent",
        side_effect=HTTPException(
            status_code=503,
            detail="Model loading failed",
        ),
    ):
        response = client.post("/generate", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model loading failed"


def test_query_endpoint_model_loading_failed():
    payload = {
        "question": "show users",
        "connection_string": "sqlite:///test.db",
    }

    with patch(
        "src.api.api_inference.get_agent",
        side_effect=HTTPException(
            status_code=503,
            detail="Model loading failed",
        ),
    ):
        response = client.post("/query", json=payload)

    assert response.status_code == 503
    assert response.json()["detail"] == "Model loading failed"
