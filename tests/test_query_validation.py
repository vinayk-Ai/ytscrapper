from fastapi.testclient import TestClient

from main import app


def test_non_dsa_question_is_blocked(monkeypatch):
    monkeypatch.setattr("main.validate_dsa_query", lambda question: False)
    monkeypatch.setattr("main.retrieve_top_results", lambda question, limit=5, video_number=None: [{
        "id": "dummy-1",
        "score": 0.99,
        "cosine_similarity": 0.99,
        "video_number": 1,
        "video_title": "Dummy lecture",
        "start": 0,
        "end": 10,
        "text": "This is a fake transcript chunk.",
    }])

    client = TestClient(app)
    response = client.post("/ask", json={"question": "capital of India"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is False
    assert "DSA" in payload["answer"]


def test_dsa_question_is_allowed(monkeypatch):
    monkeypatch.setattr("main.validate_dsa_query", lambda question: True)
    monkeypatch.setattr("main.retrieve_top_results", lambda question, limit=5, video_number=None: [{
        "id": "dummy-2",
        "score": 0.9,
        "cosine_similarity": 0.9,
        "video_number": 2,
        "video_title": "DSA lecture",
        "start": 10,
        "end": 20,
        "text": "Two pointers is a common DSA pattern.",
    }])
    monkeypatch.setattr("main.ask_llm", lambda question, context: "Two pointers uses fast and slow movement to find pairs or cycles.")

    client = TestClient(app)
    response = client.post("/ask", json={"question": "Explain two pointers in DSA"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["allowed"] is True
    assert "Two pointers" in payload["answer"]
