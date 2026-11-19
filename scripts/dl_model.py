"""DL model: a two-tower neural group recommender (PyTorch).

  item tower   per film: genre one-hot, IMDb average, vote count, Bayesian rating -> embedding
  member tower genres + rating of a member's Interested films vs their Not Interested films -> member vector
  group pool   [mean over members, min over members] -> group vector
               (a learned version of "equal weight + least misery")
  score        group . film dot product + MLP correction over [group * film, film, rating]

Trained offline on simulated movie nights (scripts/simulation.py, training seed 1000) with a ListNet + MSE
ranking loss. Unlike the RL model it is static: it does not learn from a group's own ratings.

Design notes from development:
- The recommender's 1,500 TF-IDF description dimensions are not used: they lowered NDCG@10 on the tuning
  seed, being mostly noise for a small network trained on genre-driven simulated tastes.
- Early versions fed the member tower the item tower's embeddings (or the full TF-IDF vectors); training then
  collapsed to a constant group vector that ignored the votes. A compact genre + rating member input, the
  explicit dot product and the MSE loss term fixed that.

Run from the repo root:  python -m scripts.dl_model
Registers MLflow model "group-movie-recommender-dl".
"""
import json
import os
import tempfile
import time

import numpy as np
import torch
from torch import nn

from scripts import simulation as sim_mod

CONFIG = dict(n_episodes=8000, epochs=15, batch=64, lr=1e-3, tau=0.1, loss="mixed", dropout=0.0, emb=64, hidden=256)


def input_matrix(feat):
    """Item-tower input: genre one-hot, then IMDb average /10, log votes, Bayesian rating, has-rating.
    The last two columns (Bayesian rating, has-rating) also feed the score head directly."""
    genres = feat.bandit[:, : len(feat.genres)].astype(np.float32)
    static = sim_mod.static_film_features(feat)  # bayes, avg, votes, has
    return np.hstack([genres, static[:, [1, 2, 0, 3]]]).astype(np.float32)


def member_matrix(feat):
    """Member-tower input per voted film: genre one-hot + Bayesian rating (compact taste signal)."""
    return feat.bandit[:, :-1].astype(np.float32)


class NeuralGroupRecommender(nn.Module):
    def __init__(self, d_in, d_member, emb=64, hidden=256, dropout=0.0):
        super().__init__()
        self.item = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, emb))
        self.member = nn.Sequential(nn.Linear(2 * d_member, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, emb))
        self.group = nn.Linear(2 * emb, emb)
        self.head = nn.Sequential(nn.Linear(2 * emb + 2, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1))

    def group_vector(self, voted_m, labels, member_mask):
        # voted_m (B, M, K, d_member), labels (B, M, K) 1 = Interested, member_mask (B, M)
        liked, disliked = labels.unsqueeze(-1), (1 - labels).unsqueeze(-1)
        mean_l = (voted_m * liked).sum(2) / liked.sum(2).clamp(min=1)
        mean_d = (voted_m * disliked).sum(2) / disliked.sum(2).clamp(min=1)
        m = self.member(torch.cat([mean_l, mean_d], -1))
        mask = member_mask.unsqueeze(-1)
        mean = (m * mask).sum(1) / mask.sum(1).clamp(min=1)
        least = torch.where(mask.bool(), m, torch.full_like(m, float("inf"))).min(1).values
        return self.group(torch.cat([mean, least], -1))

    def score(self, g, item_emb, item_rating):
        # g (B, emb), item_emb (B, C, emb), item_rating (B, C, 2)
        inter = g.unsqueeze(1) * item_emb
        return inter.sum(-1) + self.head(torch.cat([inter, item_emb, item_rating], -1)).squeeze(-1)

    def forward(self, voted_m, labels, member_mask, cand_x):
        g = self.group_vector(voted_m, labels, member_mask)
        return self.score(g, self.item(cand_x), cand_x[..., -2:])


def train(feat, sim, config=CONFIG, seed=sim_mod.TRAIN_SEED, log=None):
    c = config
    torch.manual_seed(seed)
    X = torch.from_numpy(input_matrix(feat))
    Mm = torch.from_numpy(member_matrix(feat))
    data = [torch.from_numpy(a) for a in sim_mod.make_episodes(c["n_episodes"], feat, sim, seed)]
    n_val = c["n_episodes"] // 10
    train_idx, val_idx = torch.arange(n_val, c["n_episodes"]), torch.arange(n_val)
    model = NeuralGroupRecommender(X.shape[1], Mm.shape[1], emb=c["emb"], hidden=c["hidden"], dropout=c["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=1e-5)

    def batch_loss(idx):
        voted, labels, mask, cand, target = (a[idx] for a in data)
        return sim_mod.ranking_loss(model(Mm[voted], labels, mask, X[cand]), target, c["tau"], c["loss"])

    history, start = [], time.perf_counter()
    for epoch in range(c["epochs"]):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        total = 0.0
        for i in range(0, len(perm), c["batch"]):
            idx = perm[i : i + c["batch"]]
            loss = batch_loss(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = batch_loss(val_idx).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(perm), "val_loss": val})
        if log:
            log(history[-1])
    return model, history, time.perf_counter() - start


class DLRanker:
    """Ranks the whole catalogue for a group with a trained network (film embeddings computed once)."""

    def __init__(self, model, feat):
        self.model, self.feat = model.eval(), feat
        self.X = torch.from_numpy(input_matrix(feat))
        self.M = torch.from_numpy(member_matrix(feat))
        with torch.no_grad():
            self.E = model.item(self.X)

    def scores(self, member_votes):
        members = [{m: v for m, v in votes.items() if m in self.feat.index} for votes in member_votes.values()]
        members = [v for v in members if v] or [{}]
        k = max(1, max(len(v) for v in members))
        # Members who voted on fewer films are padded by repeating their own votes (adds no new signal).
        voted = torch.zeros((1, len(members), k), dtype=torch.long)
        labels = torch.full((1, len(members), k), 0.5)
        for u, v in enumerate(members):
            if v:
                items = (list(v.items()) * k)[:k]
                voted[0, u] = torch.tensor([self.feat.index[m] for m, _ in items])
                labels[0, u] = torch.tensor([float(x) for _, x in items])
        with torch.no_grad():
            g = self.model.group_vector(self.M[voted], labels, torch.ones((1, len(members))))
            return self.model.score(g, self.E.unsqueeze(0), self.X[:, -2:].unsqueeze(0))[0].numpy()

    def rank(self, member_votes, history=(), rng=None, exclude=(), top_k=10):
        s = self.scores(member_votes)
        excluded = set(exclude)
        return [self.feat.ids[i] for i in np.argsort(-s, kind="stable") if self.feat.ids[i] not in excluded][:top_k]

    def describe(self):
        return {
            "family": "DL",
            "parameters": int(sum(p.numel() for p in self.model.parameters())),
            "summary": "PyTorch two-tower network (item / member towers, mean+min group pooling, dot-product score)",
        }


def load_ranker(weights_path, config_path, csv_path=sim_mod.DEFAULT_CSV):
    with open(config_path) as f:
        c = json.load(f)
    _, feat = sim_mod.load_features(csv_path)
    model = NeuralGroupRecommender(input_matrix(feat).shape[1], member_matrix(feat).shape[1], emb=c["emb"], hidden=c["hidden"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return DLRanker(model, feat)


try:
    import mlflow.pyfunc

    class DLPyfunc(mlflow.pyfunc.PythonModel):
        """Votes (member, movie_id, interested) in, ranked top 10 out."""

        def load_context(self, context):
            from scripts.dl_model import load_ranker

            a = context.artifacts
            self.ranker = load_ranker(a["weights"], a["config"], a["catalogue"])

        def predict(self, context, model_input, params=None):
            from scripts.simulation import ranked_frame, votes_from_frame

            return ranked_frame(self.ranker.feat, self.ranker.rank(votes_from_frame(model_input)))

except ImportError:
    DLPyfunc = None


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    mlflow = sim_mod.setup_mlflow(sim_mod.TRAINING_EXPERIMENT)
    from scripts import dl_model  # import by module name so MLflow pickles the class by reference

    movies, feat = sim_mod.load_features()
    sim = sim_mod.Simulator(movies, feat)
    with mlflow.start_run(run_name="dl-two-tower"):
        mlflow.set_tag("family", "DL")
        mlflow.log_params({**CONFIG, "train_seed": sim_mod.TRAIN_SEED, "framework": f"torch {torch.__version__}"})
        model, history, secs = train(feat, sim, log=lambda h: mlflow.log_metrics(
            {"train_loss": h["train_loss"], "val_loss": h["val_loss"]}, step=h["epoch"]))
        ranker = DLRanker(model, feat)
        mlflow.log_metrics({
            "train_seconds": secs, "parameters": ranker.describe()["parameters"],
            "val_ndcg_at_10": sim_mod.quick_ndcg(ranker, feat, sim),
        })
        with tempfile.TemporaryDirectory() as tmp:
            weights, config = os.path.join(tmp, "dl_weights.pt"), os.path.join(tmp, "dl_config.json")
            torch.save(model.state_dict(), weights)
            with open(config, "w") as f:
                json.dump(CONFIG, f)
            sim_mod.log_pyfunc(
                mlflow, dl_model.DLPyfunc(), feat,
                {"weights": weights, "config": config, "catalogue": sim_mod.DEFAULT_CSV},
                sim_mod.MODEL_NAMES["dl"], ["torch", "numpy", "pandas", "mlflow"],
            )
    print(f"registered {sim_mod.MODEL_NAMES['dl']}: {ranker.describe()['parameters']:,} parameters, {secs:.0f} s")


if __name__ == "__main__":
    main()
