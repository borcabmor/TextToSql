from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import src.api.api_inference as api_module
from src.api.api_inference import app

client = TestClient(app)


def test_running_endpoint():
    response = client.get("/running")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_generate_endpoint():
    mock_agent = MagicMock()
    mock_agent.generate_sql.return_value = "SELECT * FROM users"

    original_agent = api_module._agent
    api_module._agent = mock_agent

    payload = {
        "question": "show all users",
        "connection_string": "sqlite:///test.db",
    }

    response = client.post("/generate", json=payload)

    api_module._agent = original_agent

    assert response.status_code == 200
    assert response.json()["sql"] == "SELECT * FROM users"


def test_query_endpoint():
    mock_agent = MagicMock()
    mock_agent.query.return_value = {
        "sql": "SELECT * FROM users",
        "rows": [{"id": 1}],
    }

    original_agent = api_module._agent
    api_module._agent = mock_agent

    payload = {
        "question": "show all users",
        "connection_string": "sqlite:///test.db",
    }

    response = client.post("/query", json=payload)

    api_module._agent = original_agent

    assert response.status_code == 200

    data = response.json()

    assert "sql" in data
    assert "rows" in data


def test_generate_endpoint_model_not_loaded():
    original_agent = api_module._agent
    api_module._agent = None

    payload = {
        "question": "show users",
        "connection_string": "sqlite:///test.db",
    }

    response = client.post("/generate", json=payload)

    api_module._agent = original_agent

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"


def test_query_endpoint_model_not_loaded():
    original_agent = api_module._agent
    api_module._agent = None

    payload = {
        "question": "show users",
        "connection_string": "sqlite:///test.db",
    }

    response = client.post("/query", json=payload)

    api_module._agent = original_agent

    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded"
