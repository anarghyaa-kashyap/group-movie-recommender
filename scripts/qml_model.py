"""QML model: a variational quantum circuit (VQC) group recommender, hybrid quantum-classical.

For every film, classical code computes 6 features for the group (one per qubit):
  q0 mean member fit   q1 least-happy member fit   q2 happiest member fit      (content fits from votes)
  q3 Bayesian rating   q4 IMDb average             q5 log vote count            (public rating)
each scaled to 0-1 and angle-encoded as RY(pi * x).

Circuit (6 qubits, 3 layers, data re-uploading — the features are encoded again in every layer):
  repeat 3x:  RY(pi*x_q) on every qubit  ->  RY(theta) RZ(theta) on every qubit  ->  CNOT ring 0->1->...->5->0
  readout:    <Z_q> on every qubit, combined by 6 trainable classical weights into the film's score
36 quantum parameters + 6 classical readout weights, trained offline on simulated movie nights
(scripts/simulation.py, training seed 1000) with the same ListNet + MSE ranking loss as the DL model.
Like the DL model it is static: it does not learn from a group's own ratings.

The circuit is defined in Qiskit (build_qiskit_circuit) and logged to MLflow. Training and scoring use an
exact, batched statevector simulation of that same circuit in PyTorch (QuantumCircuitSim), because pushing
5,487 films x hundreds of movie nights through Qiskit one circuit at a time would take hours. main() checks
that the PyTorch simulation matches Qiskit's Statevector on random inputs before logging the model.

Run from the repo root:  python -m scripts.qml_model
Registers MLflow model "group-movie-recommender-qml". Needs qiskit (for the circuit / check) and torch.
"""
import json
import os
import tempfile
import time

import numpy as np
import torch
from torch import nn

from app import recommender as rec
from scripts import simulation as sim_mod

FEATURES = ["mean_member_fit", "min_member_fit", "max_member_fit", "bayesian_rating", "imdb_average", "log_votes"]
# 42 parameters need far less data than the DL model: 2,000 nights x 64 candidate films is plenty.
CONFIG = dict(n_qubits=len(FEATURES), layers=3, n_episodes=2000, n_cand=64, n_top=16, epochs=10, batch=32,
              lr=0.05, tau=0.1, loss="mixed")


def group_features(feat, static, member_votes, rows=None):
    """(n_films, 6) classical features for this group; static = simulation.static_film_features(feat).
    rows: only compute them for these film indices (training needs just the candidate films)."""
    sims = feat.sim if rows is None else feat.sim[rows]
    members = [v for v in member_votes.values() if v]
    penalty = rec.Weights().not_interested_penalty
    fits = []
    for votes in members:  # same member fit as app/recommender.py (member_profile + member_fit)
        profile = rec.member_profile(feat, votes, penalty)
        norm = np.linalg.norm(profile)
        fits.append(np.full(len(sims), 0.5) if norm < 1e-9 else (sims @ (profile / norm) + 1.0) / 2.0)
    fits = np.vstack(fits) if fits else np.full((1, len(sims)), 0.5)
    st = static if rows is None else static[rows]
    return np.column_stack([fits.mean(0), fits.min(0), fits.max(0), st[:, 0], st[:, 1], st[:, 2]]).astype(np.float32)


def build_qiskit_circuit(n_qubits=CONFIG["n_qubits"], layers=CONFIG["layers"]):
    """The model's circuit in Qiskit. x = encoded features (bound to pi * scaled feature), theta = trained."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    x = ParameterVector("x", n_qubits)
    theta = ParameterVector("θ", layers * n_qubits * 2)
    qc = QuantumCircuit(n_qubits, name="group_recommender_vqc")
    for layer in range(layers):
        for q in range(n_qubits):
            qc.ry(x[q], q)
        for q in range(n_qubits):
            qc.ry(theta[(layer * n_qubits + q) * 2], q)
            qc.rz(theta[(layer * n_qubits + q) * 2 + 1], q)
        for q in range(n_qubits):
            qc.cx(q, (q + 1) % n_qubits)
    return qc, x, theta


class QuantumCircuitSim(nn.Module):
    """Batched, differentiable statevector simulation of build_qiskit_circuit() (Qiskit's little-endian
    qubit order: qubit q is bit q of the basis-state index).

    Same circuit, fewer operations: per layer and qubit, RY(pi*x) RY(theta0) RZ(theta1) is applied as one
    2x2 unitary RZ(theta1) RY(pi*x + theta0) (two RYs on a qubit add up), and the 6 CNOTs of the ring are
    applied as one combined basis-state permutation. That is 7 tensor operations per layer instead of 24."""

    def __init__(self, n_qubits=CONFIG["n_qubits"], layers=CONFIG["layers"]):
        super().__init__()
        self.n, self.layers = n_qubits, layers
        self.theta = nn.Parameter(0.1 * torch.randn(layers, n_qubits, 2))
        self.readout = nn.Parameter(torch.ones(n_qubits))
        self.register_buffer("lo", torch.zeros(n_qubits))  # feature scaling, set from the training data
        self.register_buffer("hi", torch.ones(n_qubits))
        idx = torch.arange(2**n_qubits)
        ring = idx.clone()  # composing gathers: s[:, p1][:, p2] == s[:, p1[p2]]
        for q in range(n_qubits):
            ring = ring[torch.where(((idx >> q) & 1).bool(), idx ^ (1 << ((q + 1) % n_qubits)), idx)]
        self.register_buffer("cx_ring", ring, persistent=False)
        self.register_buffer("z_signs", (1 - 2 * ((idx[:, None] >> torch.arange(n_qubits)) & 1)).float(), persistent=False)

    def _apply(self, state, q, u):
        """Apply per-sample 2x2 unitaries u (B, 2, 2) to qubit q. Bit q is the middle axis of this view."""
        b = state.shape[0]
        s = state.reshape(b, 2 ** (self.n - 1 - q), 2, 2**q)
        return torch.einsum("bij,bajc->baic", u, s).reshape(b, -1)

    def scale(self, x):
        return ((x - self.lo) / (self.hi - self.lo).clamp(min=1e-6)).clamp(0, 1)

    def expectations(self, x):
        """<Z_q> for every qubit, x = raw features (B, n_qubits)."""
        angles = np.pi * self.scale(x)
        b = x.shape[0]
        state = torch.zeros((b, 2**self.n), dtype=torch.complex64, device=x.device)
        state[:, 0] = 1
        for layer in range(self.layers):
            for q in range(self.n):
                phi = angles[:, q] + self.theta[layer, q, 0]  # RY(theta0) RY(pi*x) = RY(pi*x + theta0)
                c, s = torch.cos(phi / 2), torch.sin(phi / 2)
                lam = self.theta[layer, q, 1].expand(b)
                e_minus, e_plus = torch.polar(torch.ones_like(lam), -lam / 2), torch.polar(torch.ones_like(lam), lam / 2)
                u = torch.stack([torch.stack([e_minus * c, -e_minus * s], -1),  # RZ(lam) @ RY(phi)
                                 torch.stack([e_plus * s, e_plus * c], -1)], -2)
                state = self._apply(state, q, u)
            state = state[:, self.cx_ring]
        return (state.abs() ** 2) @ self.z_signs

    def forward(self, x):
        return self.expectations(x) @ self.readout


def verify_against_qiskit(model, x, atol=1e-4):
    """Max |<Z_q>| difference between this simulation and Qiskit's Statevector on the rows of x."""
    from qiskit.quantum_info import SparsePauliOp, Statevector

    qc, xp, thp = build_qiskit_circuit(model.n, model.layers)
    theta = model.theta.detach().reshape(-1).tolist()
    with torch.no_grad():
        ours = model.expectations(x).numpy()
        scaled = model.scale(x).numpy()
    paulis = [SparsePauliOp.from_sparse_list([("Z", [q], 1.0)], num_qubits=model.n) for q in range(model.n)]
    worst = 0.0
    for i in range(len(x)):
        sv = Statevector(qc.assign_parameters({xp: (np.pi * scaled[i]).tolist(), thp: theta}))
        ref = np.array([sv.expectation_value(p).real for p in paulis])
        worst = max(worst, float(np.abs(ref - ours[i]).max()))
    if worst > atol:
        raise AssertionError(f"PyTorch simulation differs from Qiskit by {worst:.2e} (> {atol})")
    return worst


def make_training_data(feat, sim, static, config, seed):
    c = config
    print(f"simulating {c['n_episodes']} movie nights ...", flush=True)
    voted, labels, mask, cand, target = sim_mod.make_episodes(c["n_episodes"], feat, sim, seed, n_cand=c["n_cand"], n_top=c["n_top"])
    print("computing group features ...", flush=True)
    X = np.zeros((c["n_episodes"], cand.shape[1], len(FEATURES)), dtype=np.float32)
    for e in range(c["n_episodes"]):
        X[e] = group_features(feat, static, sim_mod.episode_votes(feat, voted, labels, mask, e), rows=cand[e])
    return X, target


def train(feat, sim, config=CONFIG, seed=sim_mod.TRAIN_SEED, log=None):
    c = config
    torch.manual_seed(seed)
    static = sim_mod.static_film_features(feat)
    X, target = (torch.from_numpy(a) for a in make_training_data(feat, sim, static, c, seed))
    model = QuantumCircuitSim(c["n_qubits"], c["layers"])
    flat = X.reshape(-1, X.shape[-1])
    model.lo.copy_(torch.quantile(flat, 0.01, dim=0))
    model.hi.copy_(torch.quantile(flat, 0.99, dim=0))
    n_val = c["n_episodes"] // 10
    train_idx, val_idx = torch.arange(n_val, c["n_episodes"]), torch.arange(n_val)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])

    def batch_loss(idx):
        xb = X[idx]
        scores = model(xb.reshape(-1, xb.shape[-1])).reshape(xb.shape[0], xb.shape[1])
        return sim_mod.ranking_loss(scores, target[idx], c["tau"], c["loss"])

    history, start = [], time.perf_counter()
    for epoch in range(c["epochs"]):
        perm = train_idx[torch.randperm(len(train_idx))]
        total = 0.0
        for i in range(0, len(perm), c["batch"]):
            idx = perm[i : i + c["batch"]]
            loss = batch_loss(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        with torch.no_grad():
            val = batch_loss(val_idx).item()
        history.append({"epoch": epoch + 1, "train_loss": total / len(perm), "val_loss": val})
        print(f"epoch {epoch + 1}/{c['epochs']}  train {history[-1]['train_loss']:.4f}  val {val:.4f}  "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
        if log:
            log(history[-1])
    return model, history, time.perf_counter() - start


class QMLRanker:
    def __init__(self, model, feat):
        self.model, self.feat = model.eval(), feat
        self.static = sim_mod.static_film_features(feat)

    def scores(self, member_votes):
        with torch.no_grad():
            return self.model(torch.from_numpy(group_features(self.feat, self.static, member_votes))).numpy()

    def rank(self, member_votes, history=(), rng=None, exclude=(), top_k=10):
        s = self.scores(member_votes)
        excluded = set(exclude)
        return [self.feat.ids[i] for i in np.argsort(-s, kind="stable") if self.feat.ids[i] not in excluded][:top_k]

    def describe(self):
        return {
            "family": "QML",
            "parameters": int(self.model.theta.numel() + self.model.readout.numel()),
            "summary": f"{self.model.n}-qubit variational circuit, {self.model.layers} data re-uploading layers "
            f"({self.model.theta.numel()} quantum + {self.model.readout.numel()} readout parameters)",
        }


def load_ranker(weights_path, config_path, csv_path=sim_mod.DEFAULT_CSV):
    with open(config_path) as f:
        c = json.load(f)
    _, feat = sim_mod.load_features(csv_path)
    model = QuantumCircuitSim(c["n_qubits"], c["layers"])
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return QMLRanker(model, feat)


try:
    import mlflow.pyfunc

    class QMLPyfunc(mlflow.pyfunc.PythonModel):
        """Votes (member, movie_id, interested) in, ranked top 10 out."""

        def load_context(self, context):
            from scripts.qml_model import load_ranker

            a = context.artifacts
            self.ranker = load_ranker(a["weights"], a["config"], a["catalogue"])

        def predict(self, context, model_input, params=None):
            from scripts.simulation import ranked_frame, votes_from_frame

            return ranked_frame(self.ranker.feat, self.ranker.rank(votes_from_frame(model_input)))

except ImportError:
    QMLPyfunc = None


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    mlflow = sim_mod.setup_mlflow(sim_mod.TRAINING_EXPERIMENT)
    from scripts import qml_model  # import by module name so MLflow pickles the class by reference

    movies, feat = sim_mod.load_features()
    sim = sim_mod.Simulator(movies, feat)
    with mlflow.start_run(run_name="qml-vqc"):
        mlflow.set_tag("family", "QML")
        mlflow.log_params({**CONFIG, "features": ",".join(FEATURES), "encoding": "RY(pi*x), data re-uploading",
                           "ansatz": "RY+RZ per qubit, CNOT ring", "readout": "weighted <Z> per qubit",
                           "train_seed": sim_mod.TRAIN_SEED, "simulator": "exact statevector (PyTorch), checked against Qiskit"})
        model, history, secs = train(feat, sim, log=lambda h: mlflow.log_metrics(
            {"train_loss": h["train_loss"], "val_loss": h["val_loss"]}, step=h["epoch"]))
        ranker = QMLRanker(model, feat)

        # Check the PyTorch simulation against Qiskit on 20 real (group, film) inputs.
        votes = sim_mod.sanity_votes(feat)
        rows = np.random.default_rng(0).choice(len(feat.ids), 20, replace=False)
        diff = verify_against_qiskit(model, torch.from_numpy(group_features(feat, ranker.static, votes)[rows]))
        qc, _, _ = build_qiskit_circuit(model.n, model.layers)
        mlflow.log_text(str(qc.draw(output="text", fold=-1)), "qml_circuit.txt")
        try:
            fig = qc.draw(output="mpl", fold=-1)
            mlflow.log_figure(fig, "qml_circuit.png")
        except Exception:  # matplotlib / pylatexenc not installed: the text drawing is enough
            pass

        mlflow.log_metrics({
            "train_seconds": secs, "parameters": ranker.describe()["parameters"],
            "qiskit_max_abs_diff": diff, "circuit_depth": qc.depth(),
            "val_ndcg_at_10": sim_mod.quick_ndcg(ranker, feat, sim),
        })
        with tempfile.TemporaryDirectory() as tmp:
            weights, config = os.path.join(tmp, "qml_weights.pt"), os.path.join(tmp, "qml_config.json")
            torch.save(model.state_dict(), weights)
            with open(config, "w") as f:
                json.dump(CONFIG, f)
            sim_mod.log_pyfunc(
                mlflow, qml_model.QMLPyfunc(), feat,
                {"weights": weights, "config": config, "catalogue": sim_mod.DEFAULT_CSV},
                sim_mod.MODEL_NAMES["qml"], ["torch", "qiskit", "numpy", "pandas", "mlflow"],
            )
    print(f"registered {sim_mod.MODEL_NAMES['qml']}: {ranker.describe()['summary']}, {secs:.0f} s, "
          f"matches Qiskit to {diff:.1e}")


if __name__ == "__main__":
    main()
