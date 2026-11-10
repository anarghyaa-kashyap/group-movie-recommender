import os
import sqlite3

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import connect, init_db, transaction
from .security import hash_password, new_join_code, new_token, verify_password

app = FastAPI(title="Group Movie Recommendation System", version="1.0.0")
STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.on_event("startup")
def startup():
    init_db()


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