import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from dr_code.serve.app import create_app  # noqa: E402

FENCED_TEXT = "```python\ndef add(a, b):\n    return a + b\n```"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def test_health_reports_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_profiles_lists_both_v1_profiles(client: TestClient) -> None:
    response = client.get("/profiles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["parser_version"] == "v1"
    assert set(payload["profile_ids"]) == {
        "humaneval-best-effort",
        "humaneval-field-marker",
    }


def test_explain_returns_trace(client: TestClient) -> None:
    response = client.post("/explain", json={"text": FENCED_TEXT})
    assert response.status_code == 200
    payload = response.json()
    assert payload["candidates"][0]["source"].startswith("def add")
    statuses = [candidate["status"] for candidate in payload["candidates"]]
    assert "selected" in statuses
    assert payload["rationale"]
    assert payload["roots"]


def test_explain_rejects_removed_stage_filter(client: TestClient) -> None:
    response = client.post(
        "/explain",
        json={"text": FENCED_TEXT, "stages": ["result"]},
    )
    assert response.status_code == 422


def test_explain_rejects_removed_code_field(client: TestClient) -> None:
    response = client.post(
        "/explain",
        json={"text": FENCED_TEXT, "code_field": "answer"},
    )
    assert response.status_code == 422


def test_explain_unknown_profile_is_422(client: TestClient) -> None:
    response = client.post(
        "/explain",
        json={"text": FENCED_TEXT, "profile_id": "nope"},
    )
    assert response.status_code == 422


def test_cors_allows_localhost_origins(client: TestClient) -> None:
    response = client.post(
        "/explain",
        json={"text": FENCED_TEXT},
        headers={"origin": "http://localhost:3211"},
    )
    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:3211"
    )


def test_cors_rejects_non_localhost_origins(client: TestClient) -> None:
    response = client.post(
        "/explain",
        json={"text": FENCED_TEXT},
        headers={"origin": "https://evil.example"},
    )
    assert "access-control-allow-origin" not in response.headers
