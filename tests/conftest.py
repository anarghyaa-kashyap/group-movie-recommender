import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    from app.main import app, STATE
    STATE.clear()
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    STATE.clear()


def register_login(client, name, password="password123"):
    client.post("/api/register", json={"username": name, "password": password})
    r = client.post("/api/login", json={"username": name, "password": password})
    return {"Authorization": f"Bearer {r.json()['token']}"}


def make_group(client, n=3):
    users = [register_login(client, f"user{i}") for i in range(n)]
    g = client.post("/api/groups", json={"name": "Friday", "display_name": "U0"}, headers=users[0]).json()
    for i, h in enumerate(users[1:], start=1):
        client.post("/api/groups/join", json={"code": g["join_code"], "display_name": f"U{i}"}, headers=h)
    sid = client.post(f"/api/groups/{g['group_id']}/sessions", headers=users[0]).json()["session_id"]
    return users, g, sid


def vote_all(client, headers, sid, rule=lambda m: True):
    movies = client.get(f"/api/sessions/{sid}/movies", headers=headers).json()["movies"]
    votes = [{"movie_id": m["id"], "interested": bool(rule(m))} for m in movies]
    return client.post(f"/api/sessions/{sid}/preferences", json={"votes": votes}, headers=headers), movies
