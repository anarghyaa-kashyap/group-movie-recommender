"""RL v2: the hybrid recommender with an adaptive per-group bandit (proposed improvement of rl_model.py).

Why: in the comparison, RL v1's reward stays flat over 8 movie nights — its bandit hardly learns. It has a
22-dimensional context (genres + rating + bias) but a group gives one reward per night, its weight is a fixed
0.05, and it relearns genre taste that tonight's votes already give.

What changes (base score is unchanged: S = 0.8*C + 0.2*R, with C's weight taking over when R is missing):
1. Small context x_i = [C_i, R_i, 1]. The bandit learns how much THIS group cares about matching tonight's
   votes vs public acclaim — 3 numbers, learnable from 2-3 rated nights, information the votes don't contain.
2. Residual reward: it learns (group rating scaled to 0-1) minus the base score of the watched film, i.e. what
   the base score got wrong, instead of relearning that good films get good ratings.
3. Growing influence: history weight w(n) = w_max * (1 - exp(-n / tau)) after n rated nights. A new group gets
   exactly the base ranking (cold start as in v1); the bandit counts more as evidence builds up.
4. Vectorised ranking (NumPy over the whole catalogue instead of a Python loop) — faster, same base scores.

Still per-group and isolated (FR-14), and it uses the same history format as the app, [(movie_id, group_mean)],
so no database change would be needed. w_max, tau and the prior are chosen by a grid search on tuning seed 99
(never on the evaluation seeds), as BR-06 requires.

Run from the repo root:  python -m scripts.rl_v2_model
Registers MLflow model "group-movie-recommender-rl-v2".
"""
import itertools
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass

import numpy as np

from app import recommender as rec
from scripts import simulation as sim_mod
from scripts.rl_model import RLRanker


@dataclass
class V2Config:
    w_compatibility: float = 0.8
    w_rating: float = 0.2
    w_history_max: float = 0.5
    history_tau: float = 3.0
    prior_var: float = 0.25
    noise_var: float = 0.02


# Searched on the tuning seed; 16 combinations.
GRID = {"w_history_max": [0.1, 0.25, 0.5, 1.0], "history_tau": [1.0, 3.0], "prior_var": [0.1, 0.5]}


class AdaptiveGroupBandit:
    """Thompson sampling with a Bayesian linear model of the residual reward over x = [C, R, 1]."""

    def __init__(self, prior_var, noise_var, dim=3):
        self.precision = np.eye(dim) / prior_var
        self.b = np.zeros(dim)
        self.noise_var = noise_var
        self.n = 0

    def update(self, x, y):
        self.precision += np.outer(x, x) / self.noise_var
        self.b += x * y / self.noise_var
        self.n += 1

    def sample(self, rng):
        if self.n == 0:
            return np.zeros(len(self.b))
        cov = np.linalg.inv(self.precision)
        return rng.multivariate_normal(cov @ self.b, (cov + cov.T) / 2)


class RLv2Ranker:
    def __init__(self, feat, config=None):
        self.feat, self.cfg = feat, config or V2Config()
        self.weights = rec.Weights(compatibility=self.cfg.w_compatibility, rating=self.cfg.w_rating)
        self.ids = np.array(feat.ids)
        self.has_r = np.array([r is not None for r in feat.rating])
        self.R = np.array([r if r is not None else 0.0 for r in feat.rating])

    def base_scores(self, member_votes):
        """C and the v1 base score 0.8*C + 0.2*R for every film (R missing -> C takes both weights)."""
        if any(member_votes.values()):
            C = rec.group_compatibility(self.feat, member_votes, self.weights)[0]
        else:
            C = np.zeros(len(self.ids))
        wc, wr = self.cfg.w_compatibility, self.cfg.w_rating
        return C, np.where(self.has_r, wc * C + wr * self.R, (wc + wr) * C)

    def history_weight(self, n):
        return self.cfg.w_history_max * (1.0 - np.exp(-n / self.cfg.history_tau))

    def bandit(self, history, X, base):
        bandit = AdaptiveGroupBandit(self.cfg.prior_var, self.cfg.noise_var)
        for movie_id, group_mean in history:
            i = self.feat.index.get(movie_id)
            if i is not None:
                bandit.update(X[i], (group_mean - 1.0) / 4.0 - base[i])
        return bandit

    def rank(self, member_votes, history=(), rng=None, exclude=(), top_k=10):
        rng = rng or np.random.default_rng()
        C, base = self.base_scores(member_votes)
        X = np.column_stack([C, self.R, np.ones(len(C))])
        bandit = self.bandit(history, X, base)
        score = base + self.history_weight(bandit.n) * (X @ bandit.sample(rng)) if bandit.n else base.copy()
        excluded = [self.feat.index[m] for m in exclude if m in self.feat.index]
        score[excluded] = -np.inf
        order = np.lexsort((self.ids, -score))[:top_k]  # highest score first, ties by movie id (as in v1)
        return [int(self.ids[i]) for i in order]

    def describe(self):
        c = self.cfg
        return {
            "family": "RL",
            "parameters": 3,  # bandit posterior dimension per group (learned online)
            "summary": f"content + Bayesian rating + adaptive per-group bandit over [C, R, 1] "
            f"(weights {c.w_compatibility}/{c.w_rating}, history up to {c.w_history_max}, tau {c.history_tau})",
        }


def evaluate_config(feat, sim, ranker, n_groups=15, sessions=8):
    """Tuning-seed score. 8 nights per group, because the bandit only matters after a few rated nights."""
    out = sim_mod.evaluate({"m": ranker}, feat, sim, sim_mod.TUNING_SEED, n_groups, sessions)["m"]
    return {
        "ndcg_at_10": round(float(np.mean(out["ndcg"])), 4),
        "mean_reward": round(float(out["reward"].mean()), 4),
        "reward_last_half_minus_first_half": round(float(out["reward"][sessions // 2 :].mean() - out["reward"][: sessions // 2].mean()), 4),
    }


def tune(feat, sim, log=None):
    """Grid search on the tuning seed; best = highest NDCG@10, then highest reward (as v1's weights, BR-06)."""
    keys = list(GRID)
    results = []
    for combo in itertools.product(*(GRID[k] for k in keys)):
        cfg = V2Config(**dict(zip(keys, combo)))
        res = {**dict(zip(keys, combo)), **evaluate_config(feat, sim, RLv2Ranker(feat, cfg))}
        results.append(res)
        print("  " + "  ".join(f"{k}={v}" for k, v in res.items()), flush=True)
        if log:
            log(res)
    best = max(results, key=lambda r: (r["ndcg_at_10"], r["mean_reward"]))
    return V2Config(**{k: best[k] for k in keys}), best, results


try:
    import mlflow.pyfunc

    class RLv2Pyfunc(mlflow.pyfunc.PythonModel):
        """Votes (member, movie_id, interested) in, ranked top 10 out. A cold-start ranking: pass the group's
        history to RLv2Pyfunc.ranker.rank(...) directly to include the bandit term."""

        def load_context(self, context):
            from scripts.rl_v2_model import RLv2Ranker, V2Config
            from scripts.simulation import load_features

            with open(context.artifacts["config"]) as f:
                cfg = V2Config(**json.load(f))
            _, feat = load_features(context.artifacts["catalogue"])
            self.ranker = RLv2Ranker(feat, cfg)

        def predict(self, context, model_input, params=None):
            from scripts.simulation import ranked_frame, votes_from_frame

            return ranked_frame(self.ranker.feat, self.ranker.rank(votes_from_frame(model_input)))

except ImportError:
    RLv2Pyfunc = None


def main():
    mlflow = sim_mod.setup_mlflow(sim_mod.TRAINING_EXPERIMENT)
    from scripts import rl_v2_model  # import by module name so MLflow pickles the class by reference

    movies, feat = sim_mod.load_features()
    sim = sim_mod.Simulator(movies, feat)
    with mlflow.start_run(run_name="rl-v2-adaptive-bandit"):
        mlflow.set_tag("family", "RL")
        print("RL v1 on the tuning seed (reference) ...", flush=True)
        v1 = evaluate_config(feat, sim, RLRanker(feat))
        mlflow.log_metrics({f"v1_tune_{k}": v for k, v in v1.items()})

        print(f"grid search over {np.prod([len(v) for v in GRID.values()])} settings on tuning seed "
              f"{sim_mod.TUNING_SEED} ...", flush=True)
        start = time.perf_counter()

        def log_child(res):
            with mlflow.start_run(run_name="tune " + " ".join(f"{k}={res[k]}" for k in GRID), nested=True):
                mlflow.log_params({k: res[k] for k in GRID})
                mlflow.log_metrics({k: v for k, v in res.items() if k not in GRID})

        cfg, best, _ = tune(feat, sim, log=log_child)
        secs = time.perf_counter() - start
        ranker = RLv2Ranker(feat, cfg)
        mlflow.log_params({**asdict(cfg), "context": "[C, R, 1]", "reward": "residual: (mean-1)/4 - base",
                           "history_weight": "w_max * (1 - exp(-n / tau))", "tuning_seed": sim_mod.TUNING_SEED})
        mlflow.log_metrics({"train_seconds": 0.0, "tuning_seconds": secs,
                            **{f"tune_{k}": v for k, v in best.items() if k not in GRID},
                            "val_ndcg_at_10": sim_mod.quick_ndcg(ranker, feat, sim)})
        with tempfile.TemporaryDirectory() as tmp:
            config = os.path.join(tmp, "rl_v2_config.json")
            with open(config, "w") as f:
                json.dump(asdict(cfg), f)
            sim_mod.log_pyfunc(
                mlflow, rl_v2_model.RLv2Pyfunc(), feat, {"config": config, "catalogue": sim_mod.DEFAULT_CSV},
                sim_mod.MODEL_NAMES["rl_v2"], ["numpy", "pandas", "mlflow"],
            )
    print(f"\ntuning seed  v1: NDCG {v1['ndcg_at_10']}  reward {v1['mean_reward']}  "
          f"(late - early nights {v1['reward_last_half_minus_first_half']:+})")
    print(f"tuning seed  v2: NDCG {best['ndcg_at_10']}  reward {best['mean_reward']}  "
          f"(late - early nights {best['reward_last_half_minus_first_half']:+})")
    print(f"registered {sim_mod.MODEL_NAMES['rl_v2']} with {asdict(cfg)}")


if __name__ == "__main__":
    main()
