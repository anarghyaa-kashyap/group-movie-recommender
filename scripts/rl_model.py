"""RL model: the deployed hybrid recommender (app/recommender.py) packaged for MLflow.

    S(i) = 0.8 * C(i) + 0.2 * R(i) + 0.05 * theta_g . x_i

C = content compatibility (TF-IDF + genre cosine fit, equal weight + least-misery aggregation),
R = Bayesian rating, theta_g = a sample from group g's Thompson-sampling Bayesian linear bandit,
updated from the group's 1-5 ratings of the films it watched (reinforcement learning, per group).
No offline training: the bandit learns online, and a new group starts at theta = 0 (cold start).

Run from the repo root:  python -m scripts.rl_model
Registers MLflow model "group-movie-recommender-rl".
"""
from app import recommender as rec
from scripts import simulation as sim_mod

ALL_TERMS = ("compatibility", "rating", "history")


class RLRanker:
    def __init__(self, feat, weights=None, use=ALL_TERMS):
        self.feat, self.weights, self.use = feat, weights or rec.Weights(), use

    def rank(self, member_votes, history=(), rng=None, exclude=(), top_k=10):
        ranked = rec.rank_movies(
            self.feat, member_votes, history=history, weights=self.weights, rng=rng, top_k=top_k, exclude=exclude, use=self.use
        )
        return [r["movie_id"] for r in ranked]

    def describe(self):
        w = self.weights
        return {
            "family": "RL",
            "parameters": int(self.feat.bandit.shape[1]),  # bandit posterior dimension per group (learned online)
            "summary": f"content + Bayesian rating + per-group Thompson-sampling bandit "
            f"(weights {w.compatibility}/{w.rating}/{w.history})",
        }


try:
    import mlflow.pyfunc

    class RLPyfunc(mlflow.pyfunc.PythonModel):
        """Votes (member, movie_id, interested) in, ranked top 10 out. A cold-start ranking: pass the group's
        history to RLPyfunc.ranker.rank(...) directly to include the bandit term."""

        def load_context(self, context):
            from app import recommender
            from scripts.rl_model import RLRanker
            from scripts.simulation import load_features

            _, feat = load_features(context.artifacts["catalogue"])
            self.ranker = RLRanker(feat, recommender.Weights())

        def predict(self, context, model_input, params=None):
            from scripts.simulation import ranked_frame, votes_from_frame

            return ranked_frame(self.ranker.feat, self.ranker.rank(votes_from_frame(model_input)))

except ImportError:  # mlflow is only needed for logging, not for ranking
    RLPyfunc = None


def main():
    mlflow = sim_mod.setup_mlflow(sim_mod.TRAINING_EXPERIMENT)
    from scripts import rl_model  # import by module name so MLflow pickles the class by reference

    movies, feat = sim_mod.load_features()
    sim = sim_mod.Simulator(movies, feat)
    ranker = RLRanker(feat)
    w = ranker.weights
    with mlflow.start_run(run_name="rl-hybrid"):
        mlflow.set_tag("family", "RL")
        mlflow.log_params({
            "w_compatibility": w.compatibility, "w_rating": w.rating, "w_history": w.history,
            "balance_lambda": w.balance_lambda, "not_interested_penalty": w.not_interested_penalty,
            "bandit": "Thompson sampling, Bayesian linear (prior var 0.05, noise var 0.1)",
            "bandit_context_dim": int(feat.bandit.shape[1]), "catalogue_size": len(movies),
        })
        mlflow.log_metrics({"train_seconds": 0.0, "val_ndcg_at_10": sim_mod.quick_ndcg(ranker, feat, sim)})
        sim_mod.log_pyfunc(
            mlflow, rl_model.RLPyfunc(), feat, {"catalogue": sim_mod.DEFAULT_CSV},
            sim_mod.MODEL_NAMES["rl"], ["numpy", "pandas", "mlflow"],
        )
    print(f"registered {sim_mod.MODEL_NAMES['rl']}")


if __name__ == "__main__":
    main()
