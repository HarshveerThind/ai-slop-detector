"""Creates data/videos.db with the videos table.

Run from the repo root:
    python src/init_db.py

Safe to run more than once. It will not wipe an existing table.
"""

import sqlite3
from pathlib import Path

# repo_root/data/videos.db, no matter where the script is run from
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "videos.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id             TEXT PRIMARY KEY,

    -- raw video metadata (YouTube Data API, videos.list)
    title                TEXT,
    description          TEXT,
    tags                 TEXT,       -- JSON list stored as text
    duration_seconds     INTEGER,
    view_count           INTEGER,
    like_count           INTEGER,    -- can be NULL when likes are hidden
    comment_count        INTEGER,    -- can be NULL when comments are off
    published_at         TEXT,       -- ISO 8601 timestamp
    disclosure_flag      INTEGER,    -- status.containsSyntheticMedia, 0 or 1

    -- raw channel metadata (channels.list)
    channel_id           TEXT,
    channel_title        TEXT,
    channel_created_at   TEXT,       -- ISO 8601 timestamp
    channel_video_count  INTEGER,
    channel_subs         INTEGER,    -- can be NULL when hidden

    -- file paths, relative to the repo root
    thumb_path           TEXT,
    frame1_path          TEXT,
    frame2_path          TEXT,
    frame3_path          TEXT,
    audio_path           TEXT,

    -- labels, filled in by hand, NULL until labeled
    ai_voice             INTEGER CHECK (ai_voice IN (0, 1)),
    ai_visuals           INTEGER CHECK (ai_visuals IN (0, 1)),
    label                INTEGER CHECK (label IN (0, 1)),

    -- bookkeeping
    split                TEXT CHECK (split IN ('train', 'test')),
    collected_at         TEXT        -- when the collector fetched this row
);
"""


def init_db(db_path: Path = DB_PATH, quiet: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()
    if not quiet:
        print(f"Database ready at {db_path}")


if __name__ == "__main__":
    init_db()