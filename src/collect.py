"""Collects everything for one YouTube video and stores it.

Two functions matter:

    collect_video(video_id)
        Fetches metadata, thumbnail, 3 frames and a 30 second audio clip.
        Knows nothing about labels. The live API will reuse this later.

    add_video(url, ai_voice, ai_visuals, ai_thumbnail)
        Calls collect_video, attaches your labels, inserts the row.

Use from the terminal, run from the repo root:
    python src/collect.py "https://www.youtube.com/watch?v=VIDEO_ID" 1 0 1

Check progress without adding anything:
    python src/collect.py stats

Or from Python:
    from collect import add_video
    add_video("https://youtu.be/VIDEO_ID", ai_voice=1, ai_visuals=0, ai_thumbnail=1)

Needs YOUTUBE_API_KEY in a .env file at the repo root,
plus yt-dlp and ffmpeg installed.
"""

import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv
from googleapiclient.discovery import build

from audio_utils import to_wav16k
from config import RAW_DIR as RAW, ROOT
from init_db import DB_PATH, init_db
from split import assign_splits

THUMB_DIR = RAW / "thumbs"
FRAME_DIR = RAW / "frames"
AUDIO_DIR = RAW / "audio"

CLIP_SECONDS = 30
CLIP_START = 60          # skip intros when the video is long enough
MAX_PER_CHANNEL = 5
TARGET_REAL = 150        # label = 0
TARGET_AI = 250          # label = 1

load_dotenv(ROOT / ".env")


# ---------- small helpers ----------

def extract_video_id(url: str) -> str:
    """Accepts any common YouTube link format, or a bare 11 character ID."""
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "").replace("m.", "")

    if host == "youtu.be":
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        if parsed.path == "/watch":
            candidate = parse_qs(parsed.query).get("v", [""])[0]
        else:
            # /shorts/ID, /embed/ID, /live/ID
            parts = [p for p in parsed.path.split("/") if p]
            candidate = parts[1] if len(parts) >= 2 else ""
    else:
        candidate = ""

    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        raise ValueError(f"Could not find a video ID in: {url}")
    return candidate


def parse_duration(iso: str) -> int:
    """Turns an ISO 8601 duration like PT1H2M3S into seconds."""
    m = re.fullmatch(
        r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or ""
    )
    if not m:
        return 0
    days, hours, minutes, seconds = (int(x) if x else 0 for x in m.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def to_int(value):
    return int(value) if value is not None else None


def rel(path: Path) -> str:
    """Path relative to the repo root, for storing in the DB."""
    return str(path.relative_to(ROOT))


def download(url: str, dest: Path) -> bool:
    """Downloads one file. Returns False if it does not exist."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            dest.write_bytes(resp.read())
        return True
    except urllib.error.HTTPError:
        return False


# ---------- fetching ----------

def get_client():
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise RuntimeError("YOUTUBE_API_KEY is missing. Put it in .env")
    return build("youtube", "v3", developerKey=key)


def fetch_metadata(video_id: str) -> dict:
    """Video and channel fields from the YouTube Data API. Costs 2 quota units."""
    yt = get_client()

    v = yt.videos().list(
        part="snippet,statistics,contentDetails,status", id=video_id
    ).execute()
    if not v.get("items"):
        raise ValueError(f"Video not found or not public: {video_id}")
    item = v["items"][0]
    snip, stats = item["snippet"], item.get("statistics", {})

    channel_id = snip["channelId"]
    c = yt.channels().list(part="snippet,statistics", id=channel_id).execute()
    ch = c["items"][0] if c.get("items") else {}
    ch_stats = ch.get("statistics", {})

    # best thumbnail available
    thumbs = snip.get("thumbnails", {})
    thumb_url = next(
        (thumbs[k]["url"] for k in ("maxres", "standard", "high", "medium", "default")
         if k in thumbs),
        None,
    )

    return {
        "video_id": video_id,
        "title": snip.get("title"),
        "description": snip.get("description"),
        "tags": json.dumps(snip.get("tags", [])),
        "duration_seconds": parse_duration(item["contentDetails"].get("duration")),
        "view_count": to_int(stats.get("viewCount")),
        "like_count": to_int(stats.get("likeCount")),
        "comment_count": to_int(stats.get("commentCount")),
        "published_at": snip.get("publishedAt"),
        "disclosure_flag": 1 if item.get("status", {}).get("containsSyntheticMedia") else 0,
        "channel_id": channel_id,
        "channel_title": snip.get("channelTitle"),
        "channel_created_at": ch.get("snippet", {}).get("publishedAt"),
        "channel_video_count": to_int(ch_stats.get("videoCount")),
        "channel_subs": None if ch_stats.get("hiddenSubscriberCount")
                        else to_int(ch_stats.get("subscriberCount")),
        "_thumb_url": thumb_url,   # used below, not stored
    }


def fetch_images(video_id: str, thumb_url: str | None, out_dir: Path | None = None) -> dict:
    """Custom thumbnail plus the 3 frames YouTube auto generates.

    With out_dir the images go there and the paths returned are absolute.
    Without it they go in data/raw and the paths are relative to the repo root.
    """
    paths = {"thumb_path": None, "frame1_path": None,
             "frame2_path": None, "frame3_path": None}
    thumb_dir = Path(out_dir) if out_dir else THUMB_DIR
    frame_dir = Path(out_dir) if out_dir else FRAME_DIR
    to_stored = (lambda p: str(p.resolve())) if out_dir else rel

    if thumb_url:
        dest = thumb_dir / f"{video_id}.jpg"
        if download(thumb_url, dest):
            paths["thumb_path"] = to_stored(dest)

    for n in (1, 2, 3):
        dest = frame_dir / f"{video_id}_{n}.jpg"
        if download(f"https://i.ytimg.com/vi/{video_id}/hq{n}.jpg", dest):
            paths[f"frame{n}_path"] = to_stored(dest)
        else:
            print(f"  warning, frame {n} not available")

    return paths


def fetch_audio(video_id: str, duration_seconds: int) -> str:
    """30 second clip, saved as 16 kHz mono WAV."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIO_DIR / f"{video_id}.wav"

    # long videos, skip the intro. short ones, start at 0
    start = CLIP_START if duration_seconds >= CLIP_START + CLIP_SECONDS else 0
    end = start + CLIP_SECONDS
    if duration_seconds:
        end = min(end, duration_seconds)

    with tempfile.TemporaryDirectory() as tmp:
        # step 1, yt-dlp grabs just that section of the audio stream.
        # run through this python so it works without the venv on PATH,
        # and retry because youtube sometimes fails a request at random
        for _ in range(3):
            dl = subprocess.run(
                [
                    sys.executable, "-m", "yt_dlp",
                    "--quiet", "--no-warnings", "--no-playlist",
                    "-f", "bestaudio/best",
                    "--download-sections", f"*{start}-{end}",
                    "-o", str(Path(tmp) / "clip.%(ext)s"),
                    f"https://www.youtube.com/watch?v={video_id}",
                ],
                capture_output=True, text=True,
            )
            files = list(Path(tmp).glob("clip.*"))
            if dl.returncode == 0 and files:
                break
        else:
            raise RuntimeError(f"yt-dlp failed for {video_id}: {dl.stderr.strip()}")

        # step 2, ffmpeg converts to the format the voice detector wants
        to_wav16k(files[0], out, max_seconds=CLIP_SECONDS)

    return rel(out)


def collect_video(video_id: str) -> dict:
    """Everything raw for one video. No labels. Reused by the API later."""
    row = fetch_metadata(video_id)
    thumb_url = row.pop("_thumb_url")
    row.update(fetch_images(video_id, thumb_url))
    row["audio_path"] = fetch_audio(video_id, row["duration_seconds"])
    return row


def fetch_for_scoring(video_id: str, out_dir: Path) -> dict:
    """Metadata plus image paths, no audio. The backend cannot use yt-dlp
    because YouTube blocks cloud IPs, so the browser sends the audio instead."""
    row = fetch_metadata(video_id)
    thumb_url = row.pop("_thumb_url")
    row.update(fetch_images(video_id, thumb_url, out_dir))
    return row


# ---------- storing ----------

def print_counts(conn) -> None:
    """Progress toward the dataset targets."""
    real, ai, voice, visuals = conn.execute(
        """SELECT COALESCE(SUM(label = 0), 0), COALESCE(SUM(label = 1), 0),
                  COALESCE(SUM(ai_voice = 1), 0), COALESCE(SUM(ai_visuals = 1), 0)
           FROM videos"""
    ).fetchone()
    print(f"  Real {real}/{TARGET_REAL}, AI {ai}/{TARGET_AI} "
          f"(AI voice {voice}, AI visuals {visuals})")
    if real >= TARGET_REAL:
        print("  real videos are at the cap, add AI ones from here")
    if ai >= TARGET_AI:
        print("  AI videos are at the cap, add real ones from here")


def add_video(url: str, ai_voice: int, ai_visuals: int, ai_thumbnail: int) -> None:
    """Collect one video, attach labels, insert the row.

    ai_visuals is about the video itself, ai_thumbnail about the thumbnail only.
    If the video is already in the DB, its labels are updated instead.
    """
    if any(v not in (0, 1) for v in (ai_voice, ai_visuals, ai_thumbnail)):
        raise ValueError("ai_voice, ai_visuals and ai_thumbnail must each be 0 or 1")

    video_id = extract_video_id(url)
    init_db(quiet=True)   # makes sure the table and columns exist
    label = 1 if (ai_voice or ai_visuals) else 0   # thumbnail does not count

    conn = sqlite3.connect(DB_PATH)
    try:
        exists = conn.execute(
            "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if exists:
            conn.execute(
                """UPDATE videos SET ai_voice = ?, ai_visuals = ?, ai_thumbnail = ?, label = ?
                   WHERE video_id = ?""",
                (ai_voice, ai_visuals, ai_thumbnail, label, video_id),
            )
            conn.commit()
            print(f"{video_id} was already in the DB, labels updated")
            print(f"  voice={ai_voice} visuals={ai_visuals} "
                  f"thumbnail={ai_thumbnail} label={label}")
            return

        print(f"Collecting {video_id}")
        try:
            row = collect_video(video_id)
        except Exception as e:
            # one bad video should not stop a whole batch
            print(f"  FAILED, skipped: {e}")
            return

        row["ai_voice"] = ai_voice
        row["ai_visuals"] = ai_visuals
        row["ai_thumbnail"] = ai_thumbnail
        row["label"] = label
        row["collected_at"] = datetime.now(timezone.utc).isoformat()

        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        conn.execute(f"INSERT INTO videos ({columns}) VALUES ({placeholders})", row)
        conn.commit()
        assign_splits(conn)

        count = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE channel_id = ?", (row["channel_id"],)
        ).fetchone()[0]

        print(f"Added: {row['title']}")
        print(f"  voice={ai_voice} visuals={ai_visuals} "
              f"thumbnail={ai_thumbnail} label={label}")
        print(f"  {row['channel_title']} now has {count} videos")
        print_counts(conn)
        if count > MAX_PER_CHANNEL:
            print(f"  heads up, that is over {MAX_PER_CHANNEL} from one channel")
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "stats":
        init_db(quiet=True)
        with sqlite3.connect(DB_PATH) as conn:
            print_counts(conn)
        sys.exit(0)
    if len(sys.argv) != 5:
        print('Usage: python src/collect.py "<youtube link>" '
              '<ai_voice 0|1> <ai_visuals 0|1> <ai_thumbnail 0|1>')
        sys.exit(1)
    add_video(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))