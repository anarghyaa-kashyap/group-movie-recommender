import csv
import os

from .db import connect

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "movies.csv")


def _num(value, cast):
    try:
        return cast(value) if value not in ("", None) else None
    except ValueError:
        return None


def load_catalogue(csv_path=None):
    path = csv_path or os.environ.get("MOVIES_CSV", DEFAULT_CSV)
    conn = connect()
    if conn.execute("SELECT COUNT(*) FROM movies").fetchone()[0] == 0:
        with open(path, newline="", encoding="utf-8") as f:
            rows = [
                (
                    int(r["id"]),
                    r["title"].strip(),
                    _num(r.get("year"), int),
                    r["genres"].strip(),
                    (r.get("overview") or "").strip() or None,
                    _num(r.get("vote_average"), float),
                    _num(r.get("vote_count"), int) or 0,
                )
                for r in csv.DictReader(f)
                if r.get("title") and r.get("genres")
            ]
        # One transaction for the whole catalogue: the connection autocommits, and committing
        # 5,000+ rows one by one makes startup take close to a minute.
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO movies (id, title, year, genres, overview, vote_average, vote_count) VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        conn.execute("COMMIT")
    conn.close()


def all_movies(conn):
    return [dict(r) for r in conn.execute("SELECT * FROM movies ORDER BY id")]
