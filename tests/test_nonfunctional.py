import time

import numpy as np

from app import recommender as rec
from conftest import make_group, register_login, vote_all

GENRES = ["Action", "Comedy", "Drama", "Romance", "Thriller", "Horror", "Sci-Fi", "Animation", "Family", "Mystery", "Crime", "Fantasy"]


def synthetic_catalogue(n=10000, seed=0):
    rng = np.random.default_rng(seed)
    words = "love war city night secret family friend escape journey killer dream future past island school ship king robot ghost heist".split()
    out = []
    for i in range(n):
        gs = rng.choice(GENRES, size=rng.integers(1, 4), replace=False)
        out.append({
            "id": i + 1,
            "title": f"Movie {i + 1}",
            "genres": "|".join(gs),
            "overview": " ".join(rng.choice(words, size=12)),
            "vote_average": float(np.round(rng.uniform(4, 9), 1)),
            "vote_count": int(rng.integers(10, 30000)),
        })
    return out


def test_tc17_tc_perf_01_ranking_under_3s_for_95_percent():
    feat = rec.build_features(synthetic_catalogue())
    rng = np.random.default_rng(1)
    times = []
    for run in range(20):
        members = {}
        for m in range(10):
            ids = rng.choice(feat.ids, size=10, replace=False)
            members[m] = {int(i): bool(rng.integers(0, 2)) for i in ids}
        hist = [(int(rng.choice(feat.ids)), float(rng.integers(1, 6))) for _ in range(5)]
        t = time.perf_counter()
        rec.rank_movies(feat, members, history=hist, rng=rng)
        times.append(time.perf_counter() - t)
    assert np.percentile(times, 95) < 3.0


def test_tc19_tc_grp_02_cross_group_access_denied(client):
    users, g, sid = make_group(client, 2)
    outsider = register_login(client, "mallory")
    assert client.get(f"/api/groups/{g['group_id']}", headers=outsider).status_code == 403
    assert client.get(f"/api/sessions/{sid}/status", headers=outsider).status_code == 403
    assert client.get(f"/api/sessions/{sid}/movies", headers=outsider).status_code == 403
    assert client.get("/api/groups").status_code == 401


def test_sr03_passwords_are_salted_hashes(client):
    register_login(client, "hashme", "supersecret1")
    from app.db import connect
    conn = connect()
    row = conn.execute("SELECT password_hash, salt FROM users WHERE username = 'hashme'").fetchone()
    conn.close()
    assert row["password_hash"] != "supersecret1" and len(row["salt"]) == 32


def test_sr04_injection_input_is_inert(client):
    h = register_login(client, "inj")
    r = client.post("/api/groups/join", json={"code": "' OR 1=1 --", "display_name": "x"}, headers=h)
    assert r.status_code in (404, 422)

def test_tc18_tc_rel_01_data_persists_across_restart(client):
    users, g, sid = make_group(client, 2)
    for u in users:
        vote_all(client, u, sid)
    from app.main import app, STATE
    from fastapi.testclient import TestClient
    STATE.clear()
    with TestClient(app) as restarted:
        r = restarted.get(f"/api/sessions/{sid}/recommendations", headers=users[0])
        assert r.status_code == 200 and len(r.json()["recommendations"]) == 10
    from app.db import connect
    conn = connect()
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("SELECT COUNT(*) FROM preferences").fetchone()[0] == 20
    conn.close()

def test_tc20_tc_eval_01_hybrid_beats_baselines():
    from scripts.evaluate import run
    result = run(n_groups=30, sessions=6, seed=7)
    assert result["hybrid"]["ndcg@10"] > result["rating_only"]["ndcg@10"]
    assert result["hybrid"]["ndcg@10"] >= result["preference_only"]["ndcg@10"] - 0.02