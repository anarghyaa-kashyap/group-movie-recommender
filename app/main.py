import os
import sqlite3

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np

from . import recommender
from .catalogue import all_movies, load_catalogue
from .db import connect, init_db, transaction
from .security import hash_password, new_join_code, new_token, verify_password

app = FastAPI(title="Group Movie Recommendation System", version="1.0.0")
STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
STATE = {}


def features():
    if "features" not in STATE:
        conn = connect()
        STATE["features"] = recommender.build_features(all_movies(conn))
        conn.close()
    return STATE["features"]


@app.on_event("startup")
def startup():
    init_db()
    load_catalogue()
    STATE.pop("features", None)
    features()


class ApiError(HTTPException):
    pass


@app.exception_handler(HTTPException)
async def http_error(_, exc):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_error(_, exc):
    fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
    return JSONResponse(status_code=422, content={"error": f"Invalid input: please check {', '.join(fields)}."})


@app.exception_handler(Exception)
async def unexpected_error(_, exc):
    return JSONResponse(status_code=500, content={"error": "Something went wrong and nothing was saved. Please try again."})


class Credentials(BaseModel):
    username: str = Field(min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=8, max_length=128)


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    display_name: str = Field(min_length=1, max_length=30)


class GroupJoin(BaseModel):
    code: str = Field(min_length=4, max_length=12)
    display_name: str = Field(min_length=1, max_length=30)


class Vote(BaseModel):
    movie_id: int
    interested: bool


class VoteSubmission(BaseModel):
    votes: list[Vote]


def current_user(authorization: str = Header(default="")):
    token = authorization.removeprefix("Bearer ").strip()
    if not token:
        raise HTTPException(401, "Please log in to continue.")
    conn = connect()
    row = conn.execute("SELECT user_id FROM tokens WHERE token = ?", (token,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(401, "Your session has expired. Please log in again.")
    return row["user_id"]


def require_member(conn, group_id, user_id):
    row = conn.execute("SELECT display_name FROM memberships WHERE group_id = ? AND user_id = ?", (group_id, user_id)).fetchone()
    if not row:
        raise HTTPException(403, "You are not a member of this group.")
    return row["display_name"]


def load_session(conn, session_id, user_id):
    s = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if not s:
        raise HTTPException(404, "Session not found.")
    require_member(conn, s["group_id"], user_id)
    return s


def movie_card(m):
    return {
        "id": m["id"],
        "title": m["title"],
        "year": m["year"],
        "genres": m["genres"].split("|"),
        "description": m["overview"] or "Description unavailable",
        "rating": m["vote_average"],
        "rating_display": f"{m['vote_average']:.1f}/10" if m["vote_average"] is not None else "Rating unavailable",
        "vote_count": m["vote_count"],
    }


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.post("/api/register", status_code=201)
def register(body: Credentials):
    pw_hash, salt = hash_password(body.password)
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO users (username, password_hash, salt) VALUES (?,?,?)", (body.username, pw_hash, salt))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That username is already taken.")
    return {"message": "Account created. You can now log in."}


@app.post("/api/login")
def login(body: Credentials):
    conn = connect()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (body.username,)).fetchone()
    conn.close()
    if not user or not verify_password(body.password, user["password_hash"], user["salt"]):
        raise HTTPException(401, "Invalid username or password.")
    token = new_token()
    with transaction() as conn:
        conn.execute("INSERT INTO tokens (token, user_id) VALUES (?,?)", (token, user["id"]))
    return {"token": token, "username": user["username"]}


@app.post("/api/groups", status_code=201)
def create_group(body: GroupCreate, user_id: int = Depends(current_user)):
    with transaction() as conn:
        for _ in range(10):
            code = new_join_code()
            if not conn.execute("SELECT 1 FROM groups WHERE join_code = ?", (code,)).fetchone():
                break
        cur = conn.execute("INSERT INTO groups (name, join_code, creator_id) VALUES (?,?,?)", (body.name.strip(), code, user_id))
        conn.execute("INSERT INTO memberships (group_id, user_id, display_name) VALUES (?,?,?)", (cur.lastrowid, user_id, body.display_name.strip()))
    return {"group_id": cur.lastrowid, "name": body.name.strip(), "join_code": code, "join_link": f"/?join={code}"}


@app.post("/api/groups/join")
def join_group(body: GroupJoin, user_id: int = Depends(current_user)):
    try:
        with transaction() as conn:
            g = conn.execute("SELECT * FROM groups WHERE join_code = ?", (body.code.strip().upper(),)).fetchone()
            if not g:
                raise HTTPException(404, "Invalid joining code or link.")
            if conn.execute("SELECT 1 FROM memberships WHERE group_id = ? AND user_id = ?", (g["id"], user_id)).fetchone():
                raise HTTPException(409, "You are already a member of this group.")
            conn.execute("INSERT INTO memberships (group_id, user_id, display_name) VALUES (?,?,?)", (g["id"], user_id, body.display_name.strip()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "That display name is already used in this group. Please choose another.")
    return {"group_id": g["id"], "name": g["name"]}


@app.get("/api/groups")
def my_groups(user_id: int = Depends(current_user)):
    conn = connect()
    rows = conn.execute(
        "SELECT g.id, g.name, g.join_code, m.display_name FROM groups g JOIN memberships m ON m.group_id = g.id WHERE m.user_id = ? ORDER BY g.id",
        (user_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/api/groups/{group_id}")
def group_detail(group_id: int, user_id: int = Depends(current_user)):
    conn = connect()
    require_member(conn, group_id, user_id)
    g = conn.execute("SELECT id, name, join_code FROM groups WHERE id = ?", (group_id,)).fetchone()
    members = [r["display_name"] for r in conn.execute("SELECT display_name FROM memberships WHERE group_id = ? ORDER BY display_name", (group_id,))]
    sessions = [dict(r) for r in conn.execute("SELECT id, status, watched_movie_id FROM sessions WHERE group_id = ? ORDER BY id DESC", (group_id,))]
    conn.close()
    return {**dict(g), "members": members, "sessions": sessions}


@app.post("/api/groups/{group_id}/sessions", status_code=201)
def start_session(group_id: int, user_id: int = Depends(current_user)):
    with transaction() as conn:
        require_member(conn, group_id, user_id)
        open_s = conn.execute("SELECT id FROM sessions WHERE group_id = ? AND status = 'collecting'", (group_id,)).fetchone()
        if open_s:
            return {"session_id": open_s["id"], "status": "collecting"}
        sid = conn.execute("INSERT INTO sessions (group_id) VALUES (?)", (group_id,)).lastrowid
        conn.execute("INSERT INTO session_participants (session_id, user_id) SELECT ?, user_id FROM memberships WHERE group_id = ?", (sid, group_id))
    return {"session_id": sid, "status": "collecting"}


def watched_before(conn, group_id, session_id):
    return [r["watched_movie_id"] for r in conn.execute(
        "SELECT watched_movie_id FROM sessions WHERE group_id = ? AND id < ? AND watched_movie_id IS NOT NULL", (group_id, session_id))]


@app.get("/api/sessions/{session_id}/movies")
def session_movies(session_id: int, user_id: int = Depends(current_user)):
    with transaction() as conn:
        s = load_session(conn, session_id, user_id)
        if not conn.execute("SELECT 1 FROM session_participants WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone():
            raise HTTPException(403, "You joined after this session started. Please wait for the next session.")
        rows = conn.execute("SELECT movie_id FROM presented_movies WHERE session_id = ? AND user_id = ? ORDER BY position", (session_id, user_id)).fetchall()
        ids = [r["movie_id"] for r in rows]
        if not ids:
            count = int(os.environ.get("MOVIES_PER_MEMBER", "10"))
            ids = recommender.select_movies_for_member(features(), seed=session_id * 100003 + user_id, count=count, exclude=watched_before(conn, s["group_id"], session_id))
            conn.executemany("INSERT INTO presented_movies (session_id, user_id, movie_id, position) VALUES (?,?,?,?)", [(session_id, user_id, m, i) for i, m in enumerate(ids)])
        submitted = conn.execute("SELECT submitted FROM session_participants WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()["submitted"]
    feat = features()
    return {"session_id": session_id, "locked": bool(submitted), "movies": [movie_card(feat.by_id[m]) for m in ids]}


@app.post("/api/sessions/{session_id}/preferences")
def submit_preferences(session_id: int, body: VoteSubmission, user_id: int = Depends(current_user)):
    with transaction() as conn:
        s = load_session(conn, session_id, user_id)
        if s["status"] != "collecting":
            raise HTTPException(409, "This session is closed; preferences can no longer be changed.")
        p = conn.execute("SELECT submitted FROM session_participants WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()
        if not p:
            raise HTTPException(403, "You are not a participant in this session.")
        if p["submitted"]:
            raise HTTPException(409, "Your preferences are locked for this session.")
        presented = {r["movie_id"] for r in conn.execute("SELECT movie_id FROM presented_movies WHERE session_id = ? AND user_id = ?", (session_id, user_id))}
        voted = [v.movie_id for v in body.votes]
        if not presented:
            raise HTTPException(409, "Open your movie selection before submitting.")
        if len(voted) != len(set(voted)) or set(voted) != presented:
            raise HTTPException(422, "Please mark every presented movie exactly once as Interested or Not Interested.")
        conn.executemany(
            "INSERT INTO preferences (session_id, group_id, user_id, movie_id, interested) VALUES (?,?,?,?,?)",
            [(session_id, s["group_id"], user_id, v.movie_id, int(v.interested)) for v in body.votes],
        )
        conn.execute("UPDATE session_participants SET submitted = 1 WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        pending = conn.execute("SELECT COUNT(*) FROM session_participants WHERE session_id = ? AND submitted = 0", (session_id,)).fetchone()[0]
        if pending == 0:
            generate(conn, s)
    return {"message": "Preferences saved and locked.", "waiting_for": pending}


def generate(conn, s):
    votes = {}
    for r in conn.execute("SELECT user_id, movie_id, interested FROM preferences WHERE session_id = ?", (s["id"],)):
        votes.setdefault(r["user_id"], {})[r["movie_id"]] = bool(r["interested"])
    ranked = recommender.rank_movies(
        features(),
        votes,
        weights=recommender.Weights(),
        rng=np.random.default_rng(s["id"]),
        top_k=10,
        exclude=watched_before(conn, s["group_id"], s["id"]),
    )
    conn.executemany(
        "INSERT INTO recommendations (session_id, rank, movie_id, score, compatibility, rating_score, history_score) VALUES (?,?,?,?,?,?,?)",
        [(s["id"], i + 1, r["movie_id"], r["score"], r["compatibility"], r["rating_score"], r["history_score"]) for i, r in enumerate(ranked)],
    )
    conn.execute("UPDATE sessions SET status = 'recommended' WHERE id = ?", (s["id"],))


@app.get("/api/sessions/{session_id}/status")
def session_status(session_id: int, user_id: int = Depends(current_user)):
    conn = connect()
    s = load_session(conn, session_id, user_id)
    rows = conn.execute(
        "SELECT m.display_name, p.submitted FROM session_participants p JOIN memberships m ON m.user_id = p.user_id AND m.group_id = ? WHERE p.session_id = ? ORDER BY m.display_name",
        (s["group_id"], session_id),
    ).fetchall()
    conn.close()
    return {
        "session_id": session_id,
        "status": s["status"],
        "watched_movie_id": s["watched_movie_id"],
        "submitted": [r["display_name"] for r in rows if r["submitted"]],
        "waiting_for": [r["display_name"] for r in rows if not r["submitted"]],
    }


@app.get("/api/sessions/{session_id}/recommendations")
def recommendations(session_id: int, user_id: int = Depends(current_user)):
    conn = connect()
    s = load_session(conn, session_id, user_id)
    if s["status"] == "collecting":
        conn.close()
        raise HTTPException(409, "Recommendations will be available once every member has submitted.")
    rows = conn.execute("SELECT * FROM recommendations WHERE session_id = ? ORDER BY rank", (session_id,)).fetchall()
    conn.close()
    feat = features()
    return {
        "session_id": session_id,
        "recommendations": [
            {"rank": r["rank"], **movie_card(feat.by_id[r["movie_id"]]), "match": round(r["compatibility"] * 100), "score": round(r["score"], 4)}
            for r in rows
        ],
        "note": "Recommendations balance the group's current choices with public ratings; they cannot guarantee every member will enjoy the film.",
    }
