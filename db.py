import os
import sqlite3
from contextlib import contextmanager

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS tokens (
    token TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    join_code TEXT UNIQUE NOT NULL,
    creator_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS memberships (
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    PRIMARY KEY (group_id, user_id),
    UNIQUE (group_id, display_name)
);
CREATE TABLE IF NOT EXISTS movies (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    year INTEGER,
    genres TEXT NOT NULL,
    overview TEXT,
    vote_average REAL,
    vote_count INTEGER,
    rating_source TEXT DEFAULT 'IMDb avg_vote',
    rating_scale TEXT DEFAULT '0-10'
);
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    status TEXT NOT NULL DEFAULT 'collecting',
    watched_movie_id INTEGER REFERENCES movies(id),
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS session_participants (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id),
    submitted INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, user_id)
);
CREATE TABLE IF NOT EXISTS presented_movies (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(id),
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (session_id, user_id, movie_id)
);
CREATE TABLE IF NOT EXISTS preferences (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    group_id INTEGER NOT NULL REFERENCES groups(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    interested INTEGER NOT NULL CHECK (interested IN (0, 1)),
    PRIMARY KEY (session_id, user_id, movie_id)
);
CREATE TABLE IF NOT EXISTS recommendations (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    score REAL NOT NULL,
    compatibility REAL NOT NULL,
    rating_score REAL,
    history_score REAL NOT NULL,
    PRIMARY KEY (session_id, rank)
);
CREATE TABLE IF NOT EXISTS ratings (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    group_id INTEGER NOT NULL REFERENCES groups(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    movie_id INTEGER NOT NULL REFERENCES movies(id),
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    PRIMARY KEY (session_id, user_id)
);
"""


def db_path():
    return os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "app.db"))


def connect():
    conn = sqlite3.connect(db_path(), timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction():
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(os.path.abspath(db_path())), exist_ok=True)
    conn = connect()
    conn.executescript(SCHEMA)
    conn.close()
