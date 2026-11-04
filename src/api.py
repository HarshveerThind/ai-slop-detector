"""The backend the Chrome extension talks to.

Run locally from the repo root:
    uvicorn api:app --app-dir src --port 8000

Env vars: YOUTUBE_API_KEY, MODEL_DIR, ALLOWED_ORIGINS (comma separated).
"""

import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from audio_utils import to_wav16k
from config import CACHE_DB
from predict import model_version, score_video, warm_up

VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
MAX_AUDIO_BYTES = 5 * 1024 * 1024
CACHE_MAX_AGE = timedelta(days=7)
RATE_LIMIT = 30
RATE_WINDOW_SECONDS = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    log.info("models loaded, version %s", model_version())
    yield


app = FastAPI(title="AI Slop Detector", lifespan=lifespan)


# ---------- cache ----------

def open_cache() -> sqlite3.Connection:
    CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_DB)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS scores (
               video_id      TEXT PRIMARY KEY,
               result_json   TEXT,
               model_version TEXT,
               created_at    TEXT
           )"""
    )
    return conn


def read_cache(video_id: str) -> dict | None:
    """A row only counts if it is from the loaded models and under 7 days old."""
    conn = open_cache()
    try:
        row = conn.execute(
            "SELECT result_json, model_version, created_at FROM scores WHERE video_id = ?",
            (video_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None or row[1] != model_version():
        return None
    if datetime.now(timezone.utc) - datetime.fromisoformat(row[2]) >= CACHE_MAX_AGE:
        return None
    return json.loads(row[0])


def write_cache(video_id: str, result: dict) -> None:
    conn = open_cache()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
            (video_id, json.dumps(result), model_version(),
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------- rate limit and logging ----------

request_times: dict[str, deque] = defaultdict(deque)
request_times_lock = threading.Lock()


def client_ip(request: Request) -> str:
    """Behind the AWS load balancer the real client is the last X-Forwarded-For
    entry, the one the load balancer itself added."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def over_rate_limit(ip: str) -> bool:
    now = time.monotonic()
    with request_times_lock:
        times = request_times[ip]
        while times and now - times[0] >= RATE_WINDOW_SECONDS:
            times.popleft()
        if len(times) >= RATE_LIMIT:
            return True
        times.append(now)
        # forget everyone who has gone quiet, so this does not grow forever
        if len(request_times) > 10000:
            for key in [k for k, v in request_times.items()
                        if not v or now - v[-1] >= RATE_WINDOW_SECONDS]:
                del request_times[key]
        return False


@app.middleware("http")
async def limit_and_log(request: Request, call_next):
    start = time.perf_counter()
    # /health stays open so load balancer checks never get limited
    if request.url.path.startswith("/score") and over_rate_limit(client_ip(request)):
        response = JSONResponse({"detail": "Too many requests"}, status_code=429)
    else:
        response = await call_next(request)
    log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


# added after the middleware above so it wraps it, and 429s get CORS headers too
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------- routes ----------

def check_video_id(video_id: str) -> None:
    if not VIDEO_ID.fullmatch(video_id):
        raise HTTPException(status_code=400, detail="Invalid video ID")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": model_version()}


@app.get("/score/{video_id}")
def get_score(video_id: str):
    check_video_id(video_id)
    result = read_cache(video_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No cached result")
    return result


@app.post("/score")
def post_score(video_id: str = Form(...), audio: UploadFile | None = File(None)):
    """Always scores fresh, the extension checks GET /score/{id} first. Plain def
    so FastAPI runs the slow work in a thread and the server stays responsive."""
    check_video_id(video_id)

    audio_bytes = audio.file.read(MAX_AUDIO_BYTES + 1) if audio else b""
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio is over 5 MB")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = None
        if audio_bytes:
            upload_path = Path(tmp) / "upload"
            upload_path.write_bytes(audio_bytes)
            wav_path = Path(tmp) / "clip.wav"
            try:
                to_wav16k(upload_path, wav_path)
            except RuntimeError as e:
                log.warning("bad audio for %s: %s", video_id, e)
                raise HTTPException(status_code=400, detail="Could not read the audio")

        try:
            result = score_video(video_id, wav_path)
        except ValueError as e:
            # fetch_metadata raises this for a missing or private video
            raise HTTPException(status_code=404, detail=str(e))
        except Exception:
            log.exception("scoring failed for %s", video_id)
            raise HTTPException(status_code=502, detail="Could not score this video")

    write_cache(video_id, result)
    return result
