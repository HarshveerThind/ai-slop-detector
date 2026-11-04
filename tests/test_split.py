import sqlite3

from init_db import init_db
from split import assign_splits, channel_split


def make_db(tmp_path, rows):
    path = tmp_path / "videos.db"
    init_db(path, quiet=True)
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO videos (video_id, channel_id, split) VALUES (?, ?, ?)", rows
    )
    conn.commit()
    return conn


def test_same_channel_same_split_and_stable():
    assert channel_split("UCabc") == channel_split("UCabc")
    assert channel_split("UCabc") in ("train", "test")


def test_about_a_fifth_of_channels_go_to_test():
    splits = [channel_split(f"channel{i}") for i in range(5000)]
    assert 0.17 < splits.count("test") / len(splits) < 0.23


def test_assign_fills_nulls_and_never_changes_existing(tmp_path):
    channel = "UCabc"
    flipped = "train" if channel_split(channel) == "test" else "test"
    conn = make_db(tmp_path, [
        ("aaaaaaaaaaa", channel, None),
        ("bbbbbbbbbbb", channel, None),
        ("ccccccccccc", channel, flipped),
    ])
    assert assign_splits(conn) == 2
    assert assign_splits(conn) == 0
    rows = dict(conn.execute("SELECT video_id, split FROM videos"))
    assert rows["aaaaaaaaaaa"] == rows["bbbbbbbbbbb"] == channel_split(channel)
    assert rows["ccccccccccc"] == flipped
    conn.close()
