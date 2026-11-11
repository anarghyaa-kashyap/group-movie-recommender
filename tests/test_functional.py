import sqlite3

from conftest import make_group, register_login, vote_all


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


def test_tc04_tc_pref_01_ten_movies_multiple_genres(client):
    users, g, sid = make_group(client, 1)
    movies = client.get(f"/api/sessions/{sid}/movies", headers=users[0]).json()["movies"]
    assert len(movies) == 10 and len({m["id"] for m in movies}) == 10
    assert len({genre for m in movies for genre in m["genres"]}) >= 4


def test_tc05_both_preference_values_accepted(client):
    users, g, sid = make_group(client, 1)
    r, movies = vote_all(client, users[0], sid, rule=lambda m: m["id"] % 2 == 0)
    assert r.status_code == 200


def test_tc05_incomplete_votes_rejected(client):
    users, g, sid = make_group(client, 1)
    movies = client.get(f"/api/sessions/{sid}/movies", headers=users[0]).json()["movies"]
    r = client.post(f"/api/sessions/{sid}/preferences", json={"votes": [{"movie_id": movies[0]["id"], "interested": True}]}, headers=users[0])
    assert r.status_code == 422


def test_tc06_preferences_stored_with_ids(client):
    users, g, sid = make_group(client, 1)
    vote_all(client, users[0], sid)
    from app.db import connect
    conn = connect()
    rows = conn.execute("SELECT session_id, group_id, user_id, movie_id FROM preferences").fetchall()
    conn.close()
    assert len(rows) == 10 and all(r["session_id"] == sid and r["group_id"] == g["group_id"] and r["user_id"] for r in rows)


def test_tc07_tc_sess_01_lock_and_wait_for_all(client):
    users, g, sid = make_group(client, 3)
    vote_all(client, users[0], sid)
    vote_all(client, users[1], sid)
    assert client.get(f"/api/sessions/{sid}/recommendations", headers=users[0]).status_code == 409
    again, _ = vote_all(client, users[0], sid, rule=lambda m: False)
    assert again.status_code == 409
    vote_all(client, users[2], sid)
    r = client.get(f"/api/sessions/{sid}/recommendations", headers=users[0])
    assert r.status_code == 200 and len(r.json()["recommendations"]) == 10


def test_tc12_display_has_title_description_rating(client):
    users, g, sid = make_group(client, 2)
    for u in users:
        vote_all(client, u, sid)
    recs = client.get(f"/api/sessions/{sid}/recommendations", headers=users[1]).json()["recommendations"]
    assert [r["rank"] for r in recs] == list(range(1, 11))
    assert all(r["title"] and r["description"] and r["rating_display"] for r in recs)


def finish_session(client, users, sid):
    for u in users:
        vote_all(client, u, sid)
    recs = client.get(f"/api/sessions/{sid}/recommendations", headers=users[0]).json()["recommendations"]
    client.post(f"/api/sessions/{sid}/watched", json={"movie_id": recs[0]["id"]}, headers=users[0])
    return recs


def test_tc13_rating_range(client):
    users, g, sid = make_group(client, 2)
    finish_session(client, users, sid)
    assert client.post(f"/api/sessions/{sid}/ratings", json={"rating": 6}, headers=users[0]).status_code == 422
    assert client.post(f"/api/sessions/{sid}/ratings", json={"rating": 0}, headers=users[0]).status_code == 422
    assert client.post(f"/api/sessions/{sid}/ratings", json={"rating": 1}, headers=users[0]).status_code == 201
    assert client.post(f"/api/sessions/{sid}/ratings", json={"rating": 5}, headers=users[1]).status_code == 201


def test_tc14_group_mean(client):
    users, g, sid = make_group(client, 2)
    finish_session(client, users, sid)
    client.post(f"/api/sessions/{sid}/ratings", json={"rating": 4}, headers=users[0])
    r = client.post(f"/api/sessions/{sid}/ratings", json={"rating": 2}, headers=users[1]).json()
    assert r["group_mean"] == 3.0 and r["ratings_count"] == 2


def test_tc_fb_01_mean_of_three(client):
    users, g, sid = make_group(client, 3)
    finish_session(client, users, sid)
    for u, v in zip(users, [3, 4, 5]):
        last = client.post(f"/api/sessions/{sid}/ratings", json={"rating": v}, headers=u).json()
    assert 3.99 <= last["group_mean"] <= 4.01


def test_tc15_tc_fb_02_history_influences_next_session_without_changing_old(client):
    users, g, sid = make_group(client, 2)
    first = finish_session(client, users, sid)
    for u in users:
        client.post(f"/api/sessions/{sid}/ratings", json={"rating": 5}, headers=u)
    before = client.get(f"/api/sessions/{sid}/recommendations", headers=users[0]).json()
    sid2 = client.post(f"/api/groups/{g['group_id']}/sessions", headers=users[0]).json()["session_id"]
    assert sid2 != sid
    for u in users:
        vote_all(client, u, sid2)
    from app.db import connect
    conn = connect()
    hist = conn.execute("SELECT history_score FROM recommendations WHERE session_id = ?", (sid2,)).fetchall()
    conn.close()
    assert any(abs(h["history_score"]) > 1e-9 for h in hist)
    after = client.get(f"/api/sessions/{sid}/recommendations", headers=users[0]).json()
    assert before == after
    recs2 = client.get(f"/api/sessions/{sid2}/recommendations", headers=users[0]).json()["recommendations"]
    assert first[0]["id"] not in [r["id"] for r in recs2]


def test_tc16_tc_err_01_invalid_input_and_db_failure(client, monkeypatch):
    users, g, sid = make_group(client, 1)
    r = client.post(f"/api/sessions/{sid}/preferences", json={"votes": "nope"}, headers=users[0])
    assert r.status_code == 422 and "Traceback" not in r.text
    import app.main as main
    def broken(*a, **k):
        raise sqlite3.OperationalError("disk I/O error at /secret/path")
    monkeypatch.setattr(main, "transaction", broken)
    r = client.post("/api/groups", json={"name": "X", "display_name": "Y"}, headers=users[0])
    assert r.status_code == 500 and "nothing was saved" in r.json()["error"]
    assert "secret" not in r.text and "Traceback" not in r.text


def test_watched_must_be_from_recommendations(client):
    users, g, sid = make_group(client, 1)
    vote_all(client, users[0], sid)
    r = client.post(f"/api/sessions/{sid}/watched", json={"movie_id": 999999}, headers=users[0])
    assert r.status_code == 422


def test_my_groups_lists_all_groups_for_user(client):
    a = register_login(client, "ga1")
    g1 = client.post("/api/groups", json={"name": "Group1", "display_name": "A"}, headers=a).json()
    g2 = client.post("/api/groups", json={"name": "Group2", "display_name": "A"}, headers=a).json()
    r = client.get("/api/groups", headers=a)
    assert r.status_code == 200
    assert {row["name"] for row in r.json()} == {g1["name"], g2["name"]}


def test_session_status_shows_submitted_and_waiting(client):
    users, g, sid = make_group(client, 2)
    vote_all(client, users[0], sid)
    r = client.get(f"/api/sessions/{sid}/status", headers=users[0])
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "collecting"
    assert body["submitted"] == ["U0"]
    assert body["waiting_for"] == ["U1"]


def test_feedback_endpoint_standalone(client):
    users, g, sid = make_group(client, 2)
    recs = finish_session(client, users, sid)
    client.post(f"/api/sessions/{sid}/ratings", json={"rating": 4}, headers=users[0])
    r = client.get(f"/api/sessions/{sid}/feedback", headers=users[0])
    assert r.status_code == 200
    assert r.json()["ratings_count"] == 1 and r.json()["movie_id"] == recs[0]["id"]


def test_invalid_token_rejected(client):
    r = client.get("/api/groups", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_session_not_found(client):
    a = register_login(client, "gb1")
    r = client.get("/api/sessions/999999/status", headers=a)
    assert r.status_code == 404


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


def test_start_session_returns_existing_open_session(client):
    users, g, sid = make_group(client, 2)
    r = client.post(f"/api/groups/{g['group_id']}/sessions", headers=users[0])
    assert r.status_code == 201
    assert r.json() == {"session_id": sid, "status": "collecting"}


def test_late_joiner_denied_movies(client):
    users, g, sid = make_group(client, 1)
    late = register_login(client, "late1")
    client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "Late"}, headers=late)
    r = client.get(f"/api/sessions/{sid}/movies", headers=late)
    assert r.status_code == 403


def test_submit_preferences_after_session_closed(client):
    users, g, sid = make_group(client, 1)
    r, movies = vote_all(client, users[0], sid)
    assert r.status_code == 200
    r2 = client.post(f"/api/sessions/{sid}/preferences", json={"votes": [{"movie_id": movies[0]["id"], "interested": True}]}, headers=users[0])
    assert r2.status_code == 409


def test_submit_preferences_without_opening_movies_rejected(client):
    users, g, sid = make_group(client, 1)
    r = client.post(f"/api/sessions/{sid}/preferences", json={"votes": []}, headers=users[0])
    assert r.status_code == 409


def test_submit_preferences_non_participant_rejected(client):
    users, g, sid = make_group(client, 1)
    late = register_login(client, "late2")
    client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "Late2"}, headers=late)
    r = client.post(f"/api/sessions/{sid}/preferences", json={"votes": []}, headers=late)
    assert r.status_code == 403


def test_mark_watched_while_collecting_rejected(client):
    users, g, sid = make_group(client, 1)
    r = client.post(f"/api/sessions/{sid}/watched", json={"movie_id": 1}, headers=users[0])
    assert r.status_code == 409


def test_mark_watched_twice_rejected(client):
    users, g, sid = make_group(client, 2)
    recs = finish_session(client, users, sid)
    r = client.post(f"/api/sessions/{sid}/watched", json={"movie_id": recs[0]["id"]}, headers=users[0])
    assert r.status_code == 409


def test_rate_before_watched_rejected(client):
    users, g, sid = make_group(client, 2)
    for u in users:
        vote_all(client, u, sid)
    r = client.post(f"/api/sessions/{sid}/ratings", json={"rating": 5}, headers=users[0])
    assert r.status_code == 409


def test_rate_non_participant_rejected(client):
    users, g, sid = make_group(client, 1)
    finish_session(client, users, sid)
    late = register_login(client, "late3")
    client.post("/api/groups/join", json={"code": g["join_code"], "display_name": "Late3"}, headers=late)
    r = client.post(f"/api/sessions/{sid}/ratings", json={"rating": 5}, headers=late)
    assert r.status_code == 403


def test_rate_same_movie_twice_rejected(client):
    users, g, sid = make_group(client, 2)
    finish_session(client, users, sid)
    client.post(f"/api/sessions/{sid}/ratings", json={"rating": 4}, headers=users[0])
    r = client.post(f"/api/sessions/{sid}/ratings", json={"rating": 2}, headers=users[0])
    assert r.status_code == 409
