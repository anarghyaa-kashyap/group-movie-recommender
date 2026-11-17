# Group Movie Recommendation System

SE Mini‑Project (UE24CS341A), PES University · Team: Ahana H Sharan (PES1UG24AM020), Akshata Amara (PES1UG24AM025), Anarghyaa Kashyap (PES1UG24AM039), Annanya Babu (PES1UG24AM045)

Friends join a group, each marks 10 genre‑diverse movies as **Interested / Not Interested**, and the system ranks the **whole catalogue** for the group. It balances everyone's choices (equal influence + a bonus for the least‑happy member), prefers well‑rated films (Bayesian average), and learns from the group's 1–5 rating of what they watched (a per‑group Thompson‑sampling bandit).

## Dataset

The catalogue is `data/movies.csv`: 5 487 films from **IMDb movies 2000–2020**, each with at least 10 000 votes. Columns: `id`, `title`, `year`, `genres` (pipe‑separated), `overview`, `vote_average` (0–10), `vote_count`.

To use a different catalogue with the same columns, set `MOVIES_CSV=path/to/file.csv`.

## Run it

```bash
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000. Use two browser profiles (or one normal and one incognito window) to act as two group members.

## Tests (mapped to the Test Plan and test‑case sheet)

```bash
pytest -v
python scripts/evaluate.py --groups 30 --sessions 8 --seed 7
```

| Test file | Covers |
|---|---|
| `tests/test_functional.py` | TC‑01–07, TC‑12–16, TC‑AUTH‑01, TC‑GRP‑01/02, TC‑PREF‑01, TC‑SESS‑01, TC‑FB‑01/02, TC‑ERR‑01 |
| `tests/test_recommender.py` | TC‑08–11, TC‑RECOM‑01/02, NFR‑05 (logic tested without UI/DB), NFR‑07 |
| `tests/test_nonfunctional.py` | TC‑17/TC‑PERF‑01, TC‑18/TC‑REL‑01, TC‑19, SR‑03, SR‑04, TC‑20/TC‑EVAL‑01 |
| Manual | TC‑USE‑01 (NFR‑04): `docs/usability_results.md` |

Latest results: **48/48 tests pass** · performance 20/20 requests under 3 s (p95 well under 3 s on a generated 10 000‑title catalogue, 10 members — the NFR‑01 upper bound) · NDCG@10 hybrid 0.688 vs preference‑only 0.622 vs rating‑only 0.554.

## Architecture

```
Browser (app/static/index.html)
   │  JSON over HTTP, Bearer token
FastAPI layer (app/main.py) ── auth · groups · sessions · feedback · error handling
   │                                   │
Recommender (app/recommender.py)   SQLite (app/db.py) on a volume
   features · compatibility · Bayesian rating · per‑group bandit · selection
```
Layered (presentation → API → domain → data) client–server pattern. The recommender is a pure module with no web or DB imports, as NFR‑05 requires.

Score for movie *i* in group *g*: **S(i) = 0.8·C(i) + 0.2·R(i) + 0.05·θ̃gᵀxᵢ** (weights re‑validated on the IMDb catalogue, see `docs/DEVIATIONS.md` D‑01)
- **C** = (mean member fit + 0.5 × min member fit) / 1.5
- **R** = Bayesian average rating / 10. When R is missing, its weight is reassigned to C.
- **θ̃g** = a sample from group *g*'s posterior. It is 0 for a new group (cold start).

## API

| Method & path | Purpose | FR |
|---|---|---|
| POST `/api/register`, POST `/api/login` | account, token | FR‑01 |
| POST `/api/groups`, POST `/api/groups/join`, GET `/api/groups`, GET `/api/groups/{id}` | groups, join code/link | FR‑02/03 |
| POST `/api/groups/{id}/sessions` | start or continue a movie night | FR‑07 |
| GET `/api/sessions/{id}/movies` | 10 genre‑diverse movies for this member | FR‑04 |
| POST `/api/sessions/{id}/preferences` | submit and lock votes | FR‑05/06/07 |
| GET `/api/sessions/{id}/status` | who is still voting | FR‑07 |
| GET `/api/sessions/{id}/recommendations` | ranked top 10 | FR‑08–11 |
| POST `/api/sessions/{id}/watched` | record the watched movie | FR‑12 |
| POST `/api/sessions/{id}/ratings`, GET `/api/sessions/{id}/feedback` | 1–5 rating, group mean | FR‑12/13/14 |
| GET `/healthz` | liveness/readiness probe | — |

Errors always come back as `{"error": "<readable message>"}`, with no stack traces or secrets (SR‑06, FR‑15).

## DevOps pipeline (Git · GitHub · GitHub Actions · Docker · Docker Hub · Kubernetes)

| Tool | Where |
|---|---|
| Git / GitHub | feature branch per issue → PR (template in `.github/pull_request_template.md`) → review → merge |
| GitHub Projects | backlog and sprints, created by `scripts/create_github_backlog.sh` |
| GitHub Actions | `.github/workflows/ci.yml`: runs the test suite on every push and PR. `.github/workflows/docker-publish.yml`: pushes the image to Docker Hub on `main` and on version tags |
| Docker | `Dockerfile` (non‑root, healthcheck) |
| Docker Hub | image `DOCKERHUB_USERNAME/group-movie-recommender:{latest, v1.0, sha-…}` |
| Kubernetes | `k8s/deployment.yaml` (probes, limits, `emptyDir` volume), `k8s/service.yaml` (NodePort 30080) — run locally with Minikube, no PVC needed |

### One‑time setup
1. Repo → Settings → Secrets and variables → Actions → add `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN`. Create the token at hub.docker.com → Account settings → Personal access tokens, with Read & Write access.
2. Create the backlog: `gh auth login`, then
   `GH_AKSHATA=<gh-user> GH_ANNANYA=<gh-user> GH_AHANA=<gh-user> GH_ANARGHYAA=<gh-user> scripts/create_github_backlog.sh OWNER/REPO`
3. Settings → Branches → protect `main`: require a PR, require the `CI / test` check, and require 1 approval.

### Local Docker and Kubernetes
```bash
docker build -t $DOCKERHUB_USERNAME/group-movie-recommender:latest .
docker run -p 8000:8000 $DOCKERHUB_USERNAME/group-movie-recommender:latest
docker push $DOCKERHUB_USERNAME/group-movie-recommender:latest

minikube start
sed -i "s#DOCKERHUB_USERNAME#$DOCKERHUB_USERNAME#" k8s/deployment.yaml
kubectl apply -f k8s/
kubectl get pods,svc
minikube service movie-recommender --url
```
Take a screenshot of `kubectl get pods,svc` (both running) and the `minikube service --url` page for the final demo/report — this stands in for an automated Kubernetes smoke test.

### Freeze (end of Sprint 2)
```bash
git tag -a v1.0 -m "Sprint 2 release – development frozen"
git push origin v1.0
```

## Documents
- `docs/backlog.csv`: full backlog (also loaded into the GitHub Project by `scripts/create_github_backlog.sh`)
- `docs/DEVIATIONS.md`: every difference from the SRS, SAD and Test Plan, with the reason
- `docs/evaluation_results.json`, `docs/usability_results.md`
- `videos/`: Sprint‑1 (1 min) and Sprint‑2 (2 min) recordings
