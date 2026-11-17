import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import recommender as rec

DEFAULT_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "movies.csv"))


def catalogue(n=None, csv_path=DEFAULT_CSV):
    """The real IMDb catalogue (data/movies.csv). With n, keep the n most-voted titles."""
    import csv

    with open(csv_path, newline="", encoding="utf-8") as f:
        movies = [
            {
                "id": int(r["id"]),
                "title": r["title"],
                "genres": r["genres"],
                "overview": r["overview"] or None,
                "vote_average": float(r["vote_average"]) if r["vote_average"] else None,
                "vote_count": int(r["vote_count"] or 0),
            }
            for r in csv.DictReader(f)
        ]
    if n:
        movies = sorted(movies, key=lambda m: -m["vote_count"])[:n]
    ratings = [m["vote_average"] for m in movies if m["vote_average"] is not None]
    lo, hi = min(ratings), max(ratings)
    for m in movies:
        # Hidden "quality" used to simulate how much people enjoy a film: its IMDb rating, rescaled to 0-1.
        m["_quality"] = ((m["vote_average"] if m["vote_average"] is not None else (lo + hi) / 2) - lo) / (hi - lo)
    return movies


def true_ratings(movies, tastes, genres):
    """Simulated 1-5 rating of every movie by every member: 65% genre taste, 35% film quality."""
    gidx = {g: i for i, g in enumerate(genres)}
    gm = np.zeros((len(movies), len(genres)))
    for r, m in enumerate(movies):
        for g in m["genres"].split("|"):
            gm[r, gidx[g]] = 1
    gm = gm / gm.sum(axis=1, keepdims=True)
    quality = np.array([m["_quality"] for m in movies])
    fit = tastes @ gm.T
    fit = (fit - fit.min(axis=1, keepdims=True)) / (np.ptp(fit, axis=1, keepdims=True) + 1e-9)
    return 1 + 4 * (0.65 * fit + 0.35 * quality)


def ndcg_at_10(ranked_ids, relevance, index):
    gains = [relevance[index[m]] for m in ranked_ids[:10]]
    dcg = sum((2 ** (g - 1) - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance, reverse=True)[:10]
    idcg = sum((2 ** (g - 1) - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg


METHODS = {
    "hybrid": ("compatibility", "rating", "history"),
    "preference_only": ("compatibility",),
    "rating_only": ("rating",),
}


def run(n_groups=50, sessions=10, n_movies=None, seed=0, csv_path=DEFAULT_CSV, mood=0.8):
    rng = np.random.default_rng(seed)
    movies = catalogue(n_movies, csv_path)
    feat = rec.build_features(movies)
    genres = feat.genres
    scores = {k: [] for k in METHODS}
    rewards = {k: np.zeros(sessions) for k in METHODS}
    for gi in range(n_groups):
        size = int(rng.integers(3, 6))
        base = rng.normal(0, 1, len(genres))
        tastes = base + rng.normal(0, 0.8, (size, len(genres)))
        truth = true_ratings(movies, tastes, genres)
        group_rel = truth.mean(axis=0)
        for method, use in METHODS.items():
            history, watched = [], []
            mrng = np.random.default_rng(seed * 1000 + gi)
            for s in range(sessions):
                # Votes reflect tonight's mood: each member's taste is shifted by session-specific
                # noise, while the rating after watching reflects their stable taste (truth).
                tonight = true_ratings(movies, tastes + mrng.normal(0, mood, tastes.shape), genres) if mood else truth
                votes = {}
                for u in range(size):
                    shown = rec.select_movies_for_member(feat, seed=int(mrng.integers(1e9)), count=10, exclude=watched)
                    med = np.median([tonight[u, feat.index[m]] for m in shown])
                    votes[u] = {m: bool(tonight[u, feat.index[m]] > med) for m in shown}
                ranked = rec.rank_movies(feat, votes, history=history, rng=mrng, top_k=10, exclude=watched, use=use)
                ids = [r["movie_id"] for r in ranked]
                rel = group_rel.copy()
                for w in watched:
                    rel[feat.index[w]] = 1.0
                scores[method].append(ndcg_at_10(ids, rel, feat.index))
                pick = ids[0]
                reward = float(np.clip(np.round(group_rel[feat.index[pick]] + mrng.normal(0, 0.3)), 1, 5))
                rewards[method][s] += reward / n_groups
                history.append((pick, reward))
                watched.append(pick)
    return {
        k: {"ndcg@10": round(float(np.mean(v)), 4), "mean_reward_per_session": [round(float(x), 3) for x in rewards[k]]}
        for k, v in scores.items()
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Offline evaluation of hybrid vs baselines (NFR-06 / TC-20 / TC-EVAL-01)")
    p.add_argument("--groups", type=int, default=50)
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--movies", type=int, default=None, help="keep only the N most-voted titles (default: whole catalogue)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    result = run(a.groups, a.sessions, a.movies, a.seed)
    print(json.dumps(result, indent=2))
    os.makedirs("docs", exist_ok=True)
    with open("docs/evaluation_results.json", "w") as f:
        json.dump(result, f, indent=2)
