"""Assigns train and test by channel, so one channel never spans both.

Run from the repo root:
    python src/split.py

Safe to rerun. Rows that already have a split are never changed.
"""

import hashlib
import sqlite3

from config import TEST_FRACTION
from init_db import DB_PATH, init_db

MIN_TEST_POSITIVES = 10
LABELS = ["label", "ai_voice", "ai_visuals"]


def channel_split(channel_id: str) -> str:
    bucket = int(hashlib.sha1(channel_id.encode("utf-8")).hexdigest(), 16) % 100
    return "test" if bucket < round(TEST_FRACTION * 100) else "train"


def assign_splits(conn) -> int:
    """Fills in split for every row that has none. Returns how many changed."""
    rows = conn.execute(
        "SELECT video_id, channel_id FROM videos "
        "WHERE split IS NULL AND channel_id IS NOT NULL"
    ).fetchall()
    conn.executemany(
        "UPDATE videos SET split = ? WHERE video_id = ?",
        [(channel_split(channel_id), video_id) for video_id, channel_id in rows],
    )
    conn.commit()
    return len(rows)


def print_split_table(conn) -> None:
    print(f"{'split':<7}{'videos':>8}{'channels':>10}{'label=1':>9}"
          f"{'ai_voice=1':>12}{'ai_visuals=1':>14}")
    for split in ("train", "test"):
        videos, channels, label, voice, visuals = conn.execute(
            """SELECT COUNT(*), COUNT(DISTINCT channel_id),
                      COALESCE(SUM(label = 1), 0), COALESCE(SUM(ai_voice = 1), 0),
                      COALESCE(SUM(ai_visuals = 1), 0)
               FROM videos WHERE split = ?""",
            (split,),
        ).fetchone()
        print(f"{split:<7}{videos:>8}{channels:>10}{label:>9}{voice:>12}{visuals:>14}")
        if split == "test":
            for name, count in zip(LABELS, (label, voice, visuals)):
                if count < MIN_TEST_POSITIVES:
                    print(f"WARNING: test split has only {count} positives for {name}, "
                          f"wanted at least {MIN_TEST_POSITIVES}")


if __name__ == "__main__":
    init_db(quiet=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        print(f"Assigned {assign_splits(conn)} rows")
        print_split_table(conn)
    finally:
        conn.close()
