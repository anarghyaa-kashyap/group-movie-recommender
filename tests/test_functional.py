from conftest import register_login


def test_tc01_tc_auth_01_register_login_and_reject_invalid(client):
    r = client.post("/api/register", json={"username": "alice", "password": "password123"})
    assert r.status_code == 201
    ok = client.post("/api/login", json={"username": "alice", "password": "password123"})
    assert ok.status_code == 200 and ok.json()["token"]
    bad = client.post("/api/login", json={"username": "alice", "password": "wrongpass1"})
    assert bad.status_code == 401
    assert "password123" not in bad.text and "password123" not in ok.text


def test_tc02_create_and_join_group(client):
    a, b = register_login(client, "amy1"), register_login(client, "bob1")
    g = client.post("/api/groups", json={"name": "G", "display_name": "A"}, headers=a).json()
    assert len(g["join_code"]) == 6
    r = client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "B"}, headers=b)
    assert r.status_code == 200
    assert client.get(f"/api/groups/{g['group_id']}", headers=b).json()["members"] == ["A", "B"]


def test_tc_grp_02_invalid_code_rejected(client):
    a = register_login(client, "amy2")
    r = client.post("/api/groups/join", json={"code": "ZZZZZZ", "display_name": "A"}, headers=a)
    assert r.status_code == 404 and "Invalid joining code" in r.json()["error"]


def test_tc03_tc_grp_01_duplicate_display_name_rejected(client):
    a, b = register_login(client, "amy3"), register_login(client, "bob3")
    g = client.post("/api/groups", json={"name": "G", "display_name": "Sam"}, headers=a).json()
    r = client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "Sam"}, headers=b)
    assert r.status_code == 409 and "display name" in r.json()["error"]


def test_my_groups_lists_all_groups_for_user(client):
    a = register_login(client, "ga1")
    g1 = client.post("/api/groups", json={"name": "Group1", "display_name": "A"}, headers=a).json()
    g2 = client.post("/api/groups", json={"name": "Group2", "display_name": "A"}, headers=a).json()
    r = client.get("/api/groups", headers=a)
    assert r.status_code == 200
    assert {row["name"] for row in r.json()} == {g1["name"], g2["name"]}


def test_invalid_token_rejected(client):
    r = client.get("/api/groups", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_duplicate_username_rejected(client):
    register_login(client, "dup1")
    r = client.post("/api/register", json={"username": "dup1", "password": "password123"})
    assert r.status_code == 409


def test_join_group_twice_rejected(client):
    a, b = register_login(client, "ja1"), register_login(client, "jb1")
    g = client.post("/api/groups", json={"name": "G", "display_name": "A"}, headers=a).json()
    client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "B"}, headers=b)
    r = client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "B2"}, headers=b)
    assert r.status_code == 409
