# Group Movie Recommendation System

SE Mini‑Project (UE24CS341A), PES University · Team: Ahana H Sharan (PES1UG24AM020), Akshata Amara (PES1UG24AM025), Anarghyaa Kashyap (PES1UG24AM039), Annanya Babu (PES1UG24AM045)

Friends join a group, each marks 10 genre‑diverse movies as **Interested / Not Interested**, and the system ranks the **whole catalogue** for the group. It balances everyone's choices (equal influence + a bonus for the least‑happy member), prefers well‑rated films (Bayesian average), and learns from the group's 1–5 rating of what they watched (a per‑group Thompson‑sampling bandit).

## Run it

```bash
pip install -r requirements.txt -r requirements-dev.txt
uvicorn app.main:app --reload
```
Open http://localhost:8000. Use two browser profiles (or one normal and one incognito window) to act as two group members.

The app is built over Sprint 1 (Tue 10 Nov) and Sprint 2 (Wed 11 Nov); see the GitHub Project for the backlog.
