import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

MAX_TEXT_FEATURES = 1500
SELECTION_POOL = 40

STOPWORDS = set(
    "a an the and or of to in on for with by at from as is are was be his her their its into who that this "
    "he she they them it after over must while what when where how about up out one two new".split()
)


@dataclass
class Weights:
    # Re-validated on the IMDb catalogue (scripts/evaluate.py, validation seed 1, BR-06):
    # compatibility/rating 0.4/0.6 -> 0.8/0.2 and history 0.3 -> 0.05 gave the best NDCG@10.
    compatibility: float = 0.8
    rating: float = 0.2
    history: float = 0.05
    balance_lambda: float = 0.5
    not_interested_penalty: float = 0.5


@dataclass
class Features:
    ids: list
    index: dict
    genres: list
    sim: np.ndarray
    bandit: np.ndarray
    rating: list
    by_id: dict = field(default_factory=dict)
    genre_pools: dict = field(default_factory=dict)


def _tokens(text):
    return [t for t in re.findall(r"[a-z]+", (text or "").lower()) if t not in STOPWORDS and len(t) > 2]


def bayesian_ratings(movies):
    rated = [m for m in movies if m.get("vote_average") is not None and (m.get("vote_count") or 0) > 0]
    if not rated:
        return {m["id"]: None for m in movies}
    mean = sum(m["vote_average"] for m in rated) / len(rated)
    counts = sorted(m["vote_count"] for m in rated)
    min_votes = counts[int(0.6 * (len(counts) - 1))]
    out = {}
    for m in movies:
        avg, count = m.get("vote_average"), m.get("vote_count") or 0
        if avg is None or count == 0:
            out[m["id"]] = None
        else:
            out[m["id"]] = ((count / (count + min_votes)) * avg + (min_votes / (count + min_votes)) * mean) / 10.0
    return out


def build_features(movies):
    genres = sorted({g for m in movies for g in m["genres"].split("|") if g})
    gindex = {g: i for i, g in enumerate(genres)}
    docs = [_tokens(m.get("overview")) for m in movies]
    df = Counter(t for d in docs for t in set(d))
    n = len(movies)
    # Keep the most informative terms only: drop words in more than half the descriptions,
    # then cap the vocabulary so a 5,000-10,000 title catalogue stays small in memory.
    kept = sorted((t for t, c in df.items() if c <= max(1, n // 2)), key=lambda t: (-df[t], t))[:MAX_TEXT_FEATURES]
    vocab = {t: i for i, t in enumerate(sorted(kept))}
    genre_mat = np.zeros((n, len(genres)), dtype=np.float32)
    text_mat = np.zeros((n, max(1, len(vocab))), dtype=np.float32)
    for row, (m, d) in enumerate(zip(movies, docs)):
        for g in m["genres"].split("|"):
            if g in gindex:
                genre_mat[row, gindex[g]] = 1.0
        for t, c in Counter(d).items():
            if t in vocab:
                text_mat[row, vocab[t]] = c * math.log((1 + n) / (1 + df[t]))
    genre_mat_n = genre_mat / np.maximum(np.linalg.norm(genre_mat, axis=1, keepdims=True), 1e-9)
    text_mat_n = text_mat / np.maximum(np.linalg.norm(text_mat, axis=1, keepdims=True), 1e-9)
    sim = np.hstack([genre_mat_n, 0.5 * text_mat_n])
    sim = sim / np.maximum(np.linalg.norm(sim, axis=1, keepdims=True), 1e-9)
    ratings = bayesian_ratings(movies)
    rating_list = [ratings[m["id"]] for m in movies]
    rating_col = np.array([[r if r is not None else 0.0] for r in rating_list])
    bandit = np.hstack([genre_mat.astype(np.float64), rating_col, np.ones((n, 1))])
    ids = [m["id"] for m in movies]
    pools = {g: [] for g in genres}
    for m in movies:
        for g in m["genres"].split("|"):
            if g in pools:
                pools[g].append(m["id"])
    for g, pool in pools.items():
        pool.sort(key=lambda mid: (-(ratings[mid] or 0.0), mid))
    return Features(
        ids=ids,
        index={mid: i for i, mid in enumerate(ids)},
        genres=genres,
        sim=sim,
        bandit=bandit,
        rating=rating_list,
        by_id={m["id"]: m for m in movies},
        genre_pools=pools,
    )


def member_profile(feat, votes, penalty):
    liked = [feat.index[m] for m, v in votes.items() if v and m in feat.index]
    disliked = [feat.index[m] for m, v in votes.items() if not v and m in feat.index]
    profile = np.zeros(feat.sim.shape[1])
    if liked:
        profile += feat.sim[liked].mean(axis=0)
    if disliked:
        profile -= penalty * feat.sim[disliked].mean(axis=0)
    return profile


def member_fit(feat, profile):
    norm = np.linalg.norm(profile)
    if norm < 1e-9:
        return np.full(len(feat.ids), 0.5)
    cos = feat.sim @ (profile / norm)
    return (cos + 1.0) / 2.0


def group_compatibility(feat, member_votes, weights):
    fits = np.vstack([member_fit(feat, member_profile(feat, v, weights.not_interested_penalty)) for v in member_votes.values()])
    raw = fits.mean(axis=0) + weights.balance_lambda * fits.min(axis=0)
    return raw / (1.0 + weights.balance_lambda), fits


class GroupBandit:
    def __init__(self, dim, prior_variance=0.05, noise_variance=0.1):
        self.dim = dim
        self.precision = np.eye(dim) / prior_variance
        self.b = np.zeros(dim)
        self.noise_variance = noise_variance
        self.observations = 0

    def update(self, x, reward):
        self.precision += np.outer(x, x) / self.noise_variance
        self.b += x * (reward - 0.5) / self.noise_variance
        self.observations += 1

    def sample(self, rng):
        if self.observations == 0:
            return np.zeros(self.dim)
        cov = np.linalg.inv(self.precision)
        mean = cov @ self.b
        return rng.multivariate_normal(mean, (cov + cov.T) / 2)


def bandit_from_history(feat, history):
    bandit = GroupBandit(feat.bandit.shape[1])
    for movie_id, group_mean in history:
        if movie_id in feat.index:
            bandit.update(feat.bandit[feat.index[movie_id]], (group_mean - 1.0) / 4.0)
    return bandit


def rank_movies(feat, member_votes, history=(), weights=None, rng=None, top_k=10, exclude=(), use=("compatibility", "rating", "history")):
    weights = weights or Weights()
    rng = rng or np.random.default_rng()
    n = len(feat.ids)
    compat = group_compatibility(feat, member_votes, weights)[0] if member_votes and "compatibility" in use else np.zeros(n)
    theta = bandit_from_history(feat, history).sample(rng) if "history" in use else np.zeros(feat.bandit.shape[1])
    hist = feat.bandit @ theta
    results = []
    excluded = set(exclude)
    for i, mid in enumerate(feat.ids):
        if mid in excluded:
            continue
        r = feat.rating[i]
        wc = weights.compatibility if "compatibility" in use else 0.0
        wr = weights.rating if "rating" in use else 0.0
        if r is None:
            base = compat[i] * (wc + wr) if wc else 0.0
        else:
            base = wc * compat[i] + wr * r
        score = base + weights.history * hist[i]
        results.append(
            {
                "movie_id": mid,
                "score": float(score),
                "compatibility": float(compat[i]),
                "rating_score": None if r is None else float(r),
                "history_score": float(hist[i]),
            }
        )
    results.sort(key=lambda x: (-x["score"], x["movie_id"]))
    return results[:top_k]


def select_movies_for_member(feat, seed, count=10, exclude=()):
    rng = np.random.default_rng(seed)
    excluded = set(exclude)
    pools = {}
    for g in feat.genres:
        # Draw from each genre's best-rated titles (Bayesian average), so members see
        # recognisable, well-rated films while different members still get different picks.
        top = [mid for mid in feat.genre_pools[g] if mid not in excluded][: max(count, SELECTION_POOL)]
        rng.shuffle(top)
        pools[g] = top
    # Genre order is random but weighted by genre size, so common genres (Drama, Comedy,
    # Action) appear more often than rare ones (Western, Musical) while every genre can appear.
    sizes = np.array([len(feat.genre_pools[g]) for g in feat.genres], dtype=float)
    order = [str(g) for g in rng.choice(feat.genres, size=len(feat.genres), replace=False, p=sizes / sizes.sum())] if sizes.sum() else []
    chosen = []
    while len(chosen) < count and any(pools.values()):
        for g in order:
            while pools[g] and pools[g][0] in chosen:
                pools[g].pop(0)
            if pools[g]:
                chosen.append(pools[g].pop(0))
            if len(chosen) == count:
                break
    return chosen
