"""Collects everything for one YouTube video and stores it.

Two functions matter:

    collect_video(video_id)
        Fetches metadata, thumbnail, 3 frames and a 30 second audio clip.
        Knows nothing about labels. The live API will reuse this later.

    add_video(url, ai_voice, ai_visuals)
        Calls collect_video, attaches your labels, inserts the row.

Use from the terminal, run from the repo root:
    python src/collect.py "https://www.youtube.com/watch?v=VIDEO_ID" 1 0

Check progress without adding anything:
    python src/collect.py stats

Or from Python:
    from collect import add_video
    add_video("https://youtu.be/VIDEO_ID", ai_voice=1, ai_visuals=0)

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

from init_db import DB_PATH, init_db

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
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


def fetch_images(video_id: str, thumb_url: str | None) -> dict:
    """Custom thumbnail plus the 3 frames YouTube auto generates."""
    paths = {"thumb_path": None, "frame1_path": None,
             "frame2_path": None, "frame3_path": None}

    if thumb_url:
        dest = THUMB_DIR / f"{video_id}.jpg"
        if download(thumb_url, dest):
            paths["thumb_path"] = rel(dest)

    for n in (1, 2, 3):
        dest = FRAME_DIR / f"{video_id}_{n}.jpg"
        if download(f"https://i.ytimg.com/vi/{video_id}/hq{n}.jpg", dest):
            paths[f"frame{n}_path"] = rel(dest)
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
        # step 1, yt-dlp grabs just that section of the audio stream
        dl = subprocess.run(
            [
                "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
                "-f", "bestaudio/best",
                "--download-sections", f"*{start}-{end}",
                "-o", str(Path(tmp) / "clip.%(ext)s"),
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True,
        )
        files = list(Path(tmp).glob("clip.*"))
        if dl.returncode != 0 or not files:
            raise RuntimeError(f"yt-dlp failed for {video_id}: {dl.stderr.strip()}")

        # step 2, ffmpeg converts to the format the voice detector wants
        ff = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(files[0]),
                "-ar", "16000", "-ac", "1",
                "-t", str(CLIP_SECONDS),
                str(out),
            ],
            capture_output=True, text=True,
        )
        if ff.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {video_id}: {ff.stderr.strip()}")

    return rel(out)


def collect_video(video_id: str) -> dict:
    """Everything raw for one video. No labels. Reused by the API later."""
    row = fetch_metadata(video_id)
    thumb_url = row.pop("_thumb_url")
    row.update(fetch_images(video_id, thumb_url))
    row["audio_path"] = fetch_audio(video_id, row["duration_seconds"])
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


def add_video(url: str, ai_voice: int, ai_visuals: int) -> None:
    """Collect one video, attach labels, insert the row."""
    if ai_voice not in (0, 1) or ai_visuals not in (0, 1):
        raise ValueError("ai_voice and ai_visuals must each be 0 or 1")

    video_id = extract_video_id(url)
    init_db(quiet=True)   # makes sure the table exists, does nothing if it does

    conn = sqlite3.connect(DB_PATH)
    try:
        exists = conn.execute(
            "SELECT 1 FROM videos WHERE video_id = ?", (video_id,)
        ).fetchone()
        if exists:
            print(f"{video_id} is already in the DB, skipped")
            return

        print(f"Collecting {video_id}")
        row = collect_video(video_id)

        row["ai_voice"] = ai_voice
        row["ai_visuals"] = ai_visuals
        row["label"] = 1 if (ai_voice or ai_visuals) else 0
        row["collected_at"] = datetime.now(timezone.utc).isoformat()

        columns = ", ".join(row)
        placeholders = ", ".join(f":{k}" for k in row)
        conn.execute(f"INSERT INTO videos ({columns}) VALUES ({placeholders})", row)
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE channel_id = ?", (row["channel_id"],)
        ).fetchone()[0]

        print(f"Added: {row['title']}")
        print(f"  voice={ai_voice} visuals={ai_visuals} label={row['label']}")
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
    if len(sys.argv) != 4:
        print('Usage: python src/collect.py "<youtube link>" <ai_voice 0|1> <ai_visuals 0|1>')
        sys.exit(1)
    add_video(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))