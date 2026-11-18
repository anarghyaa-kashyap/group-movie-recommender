"""Shared pieces for the RL, DL and QML model scripts.

- the IMDb catalogue and the recommender's features
- the simulated groups from scripts/evaluate.py (65 % genre taste + 35 % film quality, votes affected by
  tonight's mood), used to make training data for DL/QML and to evaluate all three models
- the NFR-06 evaluation protocol (NDCG@10 + reward per movie night), with any model plugged in
- MLflow setup and the pyfunc input format (votes in, ranked top 10 out)

Every model exposes the same ranker interface:
    ranker.rank(member_votes, history=(), rng=None, exclude=(), top_k=10) -> [movie_id, ...]
where member_votes = {member: {movie_id: interested_bool}} and history = [(movie_id, group_mean_rating), ...].
"""
import csv
import os
import time

import numpy as np

from app import recommender as rec
from scripts.evaluate import DEFAULT_CSV, ndcg_at_10

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_SEED = 1000  # training data for DL / QML
TUNING_SEED = 99  # validation while choosing settings
EVAL_SEEDS = (7, 11, 23)  # final comparison (seed 7 is the NFR-06 seed); disjoint from the two above
N_GROUPS, SESSIONS = 30, 8

TRAINING_EXPERIMENT = "group-recommender-models"
COMPARISON_EXPERIMENT = "group-recommender-comparison"
MODEL_NAMES = {
    "rl": "group-movie-recommender-rl",
    "rl_v2": "group-movie-recommender-rl-v2",
    "dl": "group-movie-recommender-dl",
    "qml": "group-movie-recommender-qml",
}


# ---------------------------------------------------------------- catalogue

def load_movies(csv_path=DEFAULT_CSV):
    """The IMDb catalogue, plus the hidden 0-1 "quality" the simulator uses (as in scripts/evaluate.py)."""
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
    ratings = [m["vote_average"] for m in movies if m["vote_average"] is not None]
    lo, hi = min(ratings), max(ratings)
    for m in movies:
        m["_quality"] = ((m["vote_average"] if m["vote_average"] is not None else (lo + hi) / 2) - lo) / (hi - lo)
    return movies


def load_features(csv_path=DEFAULT_CSV):
    movies = load_movies(csv_path)
    return movies, rec.build_features(movies)


def static_film_features(feat):
    """Per film: Bayesian rating, IMDb average / 10, log vote count scaled to 0-1, has-rating flag."""
    movies = [feat.by_id[m] for m in feat.ids]
    bayes = np.array([x if x is not None else 0.0 for x in feat.rating], dtype=np.float32)
    avg = np.array([(m["vote_average"] or 0.0) / 10 for m in movies], dtype=np.float32)
    votes = np.log1p(np.array([m["vote_count"] or 0 for m in movies], dtype=np.float32))
    votes = votes / max(float(votes.max()), 1.0)
    has = np.array([x is not None for x in feat.rating], dtype=np.float32)
    return np.column_stack([bayes, avg, votes, has])


# ---------------------------------------------------------------- simulator

class Simulator:
    """Vectorised copy of scripts/evaluate.py's true_ratings (65 % genre taste + 35 % film quality)."""

    def __init__(self, movies, feat):
        gidx = {g: i for i, g in enumerate(feat.genres)}
        gm = np.zeros((len(movies), len(feat.genres)))
        for row, m in enumerate(movies):
            for g in m["genres"].split("|"):
                gm[row, gidx[g]] = 1
        self.gm_t = (gm / gm.sum(axis=1, keepdims=True)).T
        self.quality = np.array([m["_quality"] for m in movies])

    def true_ratings(self, tastes):
        fit = tastes @ self.gm_t
        fit = (fit - fit.min(axis=1, keepdims=True)) / (np.ptp(fit, axis=1, keepdims=True) + 1e-9)
        return 1 + 4 * (0.65 * fit + 0.35 * self.quality)


def make_episodes(n, feat, sim, seed, k=10, max_members=5, n_cand=128, n_top=24, mood=0.8):
    """Simulated movie nights for training: what each member saw, how they voted, and the group's true
    mean rating of a set of candidate films (the group's top n_top films + random others)."""
    rng = np.random.default_rng(seed)
    G = len(feat.genres)
    voted = np.zeros((n, max_members, k), dtype=np.int64)
    labels = np.zeros((n, max_members, k), dtype=np.float32)
    mask = np.zeros((n, max_members), dtype=np.float32)
    cand = np.zeros((n, n_cand), dtype=np.int64)
    target = np.zeros((n, n_cand), dtype=np.float32)
    n_items = len(feat.ids)
    for e in range(n):
        size = int(rng.integers(3, 6))
        tastes = rng.normal(0, 1, G) + rng.normal(0, 0.8, (size, G))
        group_rel = sim.true_ratings(tastes).mean(axis=0)
        tonight = sim.true_ratings(tastes + rng.normal(0, mood, tastes.shape))
        for u in range(size):
            shown = [feat.index[m] for m in rec.select_movies_for_member(feat, seed=int(rng.integers(1e9)), count=k)]
            med = np.median(tonight[u, shown])
            voted[e, u] = shown
            labels[e, u] = tonight[u, shown] > med
            mask[e, u] = 1
        top = np.argsort(-group_rel)[:n_top]
        rest = rng.choice(np.setdiff1d(np.arange(n_items), top), n_cand - n_top, replace=False)
        cand[e] = np.concatenate([top, rest])
        target[e] = group_rel[cand[e]]
    return voted, labels, mask, cand, target


def episode_votes(feat, voted, labels, mask, e):
    """One training episode as the {member: {movie_id: interested}} dict the rankers take."""
    return {
        u: {feat.ids[int(voted[e, u, j])]: bool(labels[e, u, j]) for j in range(voted.shape[2])}
        for u in range(voted.shape[1])
        if mask[e, u]
    }


def ranking_loss(scores, target, tau=0.1, kind="mixed"):
    """ListNet (softmax cross-entropy against the true group rating) and/or MSE on the standardised target.
    ListNet alone focuses on the few top films, which differ mostly by public rating; the MSE term makes the
    model also order the rest of the list, which is where the members' votes matter."""
    import torch

    listnet = -(torch.softmax(target / tau, -1) * torch.log_softmax(scores, -1)).sum(-1).mean()
    if kind == "listnet":
        return listnet
    z = (target - target.mean(-1, keepdim=True)) / (target.std(-1, keepdim=True) + 1e-6)
    mse = ((scores - scores.mean(-1, keepdim=True)) - z).pow(2).mean()
    return mse if kind == "mse" else listnet + mse


# ---------------------------------------------------------------- evaluation protocol

def evaluate(rankers, feat, sim, seed, n_groups=N_GROUPS, sessions=SESSIONS, mood=0.8):
    """The NFR-06 protocol of scripts/evaluate.run(), with any rankers plugged in.

    For the RL hybrid this reproduces scripts/evaluate.py exactly (same random streams).
    Extra metrics: Precision@10 (share of the top 10 in the group's true top 5 %), ranking latency, and the
    median IMDb vote count of the recommended films (how mainstream the recommendations are).
    """
    rng = np.random.default_rng(seed)
    G = len(feat.genres)
    out = {k: {"ndcg": [], "p10": [], "reward": np.zeros(sessions), "ms": [], "votes": []} for k in rankers}
    for gi in range(n_groups):
        size = int(rng.integers(3, 6))
        tastes = rng.normal(0, 1, G) + rng.normal(0, 0.8, (size, G))
        group_rel = sim.true_ratings(tastes).mean(axis=0)
        top5 = group_rel >= np.quantile(group_rel, 0.95)
        for name, ranker in rankers.items():
            history, watched = [], []
            mrng = np.random.default_rng(seed * 1000 + gi)
            for s in range(sessions):
                tonight = sim.true_ratings(tastes + mrng.normal(0, mood, tastes.shape))
                votes = {}
                for u in range(size):
                    shown = rec.select_movies_for_member(feat, seed=int(mrng.integers(1e9)), count=10, exclude=watched)
                    med = np.median([tonight[u, feat.index[m]] for m in shown])
                    votes[u] = {m: bool(tonight[u, feat.index[m]] > med) for m in shown}
                t = time.perf_counter()
                ids = ranker.rank(votes, history=list(history), rng=mrng, exclude=list(watched), top_k=10)
                out[name]["ms"].append((time.perf_counter() - t) * 1000)
                rel = group_rel.copy()
                for w in watched:
                    rel[feat.index[w]] = 1.0
                out[name]["ndcg"].append(ndcg_at_10(ids, rel, feat.index))
                out[name]["p10"].append(float(np.mean([top5[feat.index[m]] for m in ids])))
                out[name]["votes"].append(float(np.median([feat.by_id[m]["vote_count"] for m in ids])))
                pick = ids[0]
                reward = float(np.clip(np.round(group_rel[feat.index[pick]] + mrng.normal(0, 0.3)), 1, 5))
                out[name]["reward"][s] += reward / n_groups
                history.append((pick, reward))
                watched.append(pick)
    return out


def summarise(per_seed):
    """Average evaluate() results over seeds."""
    summary = {}
    for n in per_seed[0]:
        ndcg = [float(np.mean(r[n]["ndcg"])) for r in per_seed]
        reward = np.mean([r[n]["reward"] for r in per_seed], axis=0)
        ms = np.concatenate([r[n]["ms"] for r in per_seed])
        summary[n] = {
            "ndcg@10": round(float(np.mean(ndcg)), 4),
            "ndcg@10_per_seed": [round(x, 4) for x in ndcg],
            "precision@10": round(float(np.mean([np.mean(r[n]["p10"]) for r in per_seed])), 4),
            "mean_reward": round(float(reward.mean()), 3),
            "reward_per_session": [round(float(x), 3) for x in reward],
            "latency_ms_mean": round(float(ms.mean()), 2),
            "latency_ms_p95": round(float(np.percentile(ms, 95)), 2),
            "median_imdb_votes_top10": int(np.median(np.concatenate([r[n]["votes"] for r in per_seed]))),
        }
    return summary


def quick_ndcg(ranker, feat, sim, seed=TUNING_SEED, n_groups=12, sessions=2):
    """Small validation score each model script logs after training (on the tuning seed, not an eval seed)."""
    return round(float(np.mean(evaluate({"m": ranker}, feat, sim, seed, n_groups, sessions)["m"]["ndcg"])), 4)


def sanity_votes(feat, likes=("Animation", "Family", "Comedy", "Adventure"), members=3):
    """A hand-made group: each member votes on 10 genre-diverse films and likes the ones in `likes`."""
    votes = {}
    for u in range(members):
        shown = rec.select_movies_for_member(feat, seed=100 + u, count=10)
        votes[u] = {m: any(g in likes for g in feat.by_id[m]["genres"].split("|")) for m in shown}
    return votes


# ---------------------------------------------------------------- MLflow

def setup_mlflow(experiment):
    """Local SQLite tracking store in the repo root (override with MLFLOW_TRACKING_URI)."""
    import mlflow

    uri = os.environ.get("MLFLOW_TRACKING_URI") or "sqlite:///" + os.path.join(ROOT, "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return mlflow


def votes_from_frame(model_input):
    """pyfunc input (one row per vote: member, movie_id, interested) -> {member: {movie_id: interested}}."""
    votes = {}
    for r in model_input.itertuples():
        votes.setdefault(str(r.member), {})[int(r.movie_id)] = bool(r.interested)
    return votes


def ranked_frame(feat, ids):
    import pandas as pd

    return pd.DataFrame([{"rank": i + 1, "movie_id": m, "title": feat.by_id[m]["title"]} for i, m in enumerate(ids)])


def example_input(feat):
    import pandas as pd

    votes = sanity_votes(feat, members=2)
    return pd.DataFrame([{"member": str(u), "movie_id": m, "interested": v} for u, mv in votes.items() for m, v in mv.items()])


def log_pyfunc(mlflow, python_model, feat, artifacts, registered_name, pip_requirements):
    """Log a ranker as an MLflow pyfunc model (votes in, top 10 out) and register it."""
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=python_model,
        artifacts=artifacts,
        code_paths=[os.path.join(ROOT, "app"), os.path.join(ROOT, "scripts")],
        input_example=example_input(feat),
        pip_requirements=pip_requirements,
        registered_model_name=registered_name,
    )
