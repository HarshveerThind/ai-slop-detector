"""The feature code. Training and serving both use these functions, so a feature
can never mean one thing in training and another in the API.

Build the caches used for training, from the repo root:
    python src/features.py
"""

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AI_TERMS, CLIP_MODEL, CLIP_PRETRAINED, FEATURE_DIR, ROOT, SAMPLE_RATE,
)
from init_db import DB_PATH

NAN = float("nan")
EMBEDDING_SIZE = 512
EMBEDDING_COLUMNS = [f"e{i}" for i in range(EMBEDDING_SIZE)]
VOICE_WINDOW_SECONDS = 5
MIN_WINDOW_SECONDS = 1   # a shorter tail than this is not worth scoring
SILENCE_PEAK = 1e-4

FRAME_COLUMNS = ["frame1_path", "frame2_path", "frame3_path"]
IMAGE_COLUMNS = ["thumb_path"] + FRAME_COLUMNS

METADATA_FEATURES = [
    "title_len", "title_word_count", "title_caps_ratio", "title_exclaim",
    "title_question", "title_digit_count", "title_emoji_count",
    "desc_len", "desc_link_count", "desc_hashtag_count", "desc_ai_terms",
    "tag_count", "duration_seconds", "is_short", "log_views", "like_per_view",
    "comment_per_view", "video_age_days", "views_per_day", "disclosure_flag",
    "channel_age_days", "channel_video_count", "uploads_per_day",
    "log_channel_subs", "subs_per_video",
]

EMOJI = re.compile(
    "[\U0001F1E6-\U0001F1FF\U0001F300-\U0001FAFF☀-➿⭐⭕]"
)
HASHTAG = re.compile(r"#\w+")
AI_TERMS_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(t).replace(r"\ ", r"[\s-]+") for t in AI_TERMS) + r")\b",
    re.IGNORECASE,
)


# ---------- small helpers ----------

def stored_path(value: str | None) -> Path | None:
    """Turns a path from the DB into a real path. DB paths are relative to the
    repo root and may have been written on Windows."""
    if not value:
        return None
    path = Path(value.replace("\\", "/"))
    return path if path.is_absolute() else ROOT / path


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def load_train_rows() -> list[dict]:
    """Labeled train rows only. Nothing here can see the test split."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM videos WHERE split = 'train' AND label IS NOT NULL "
            "ORDER BY video_id"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def _number(value) -> float:
    return NAN if value is None else float(value)


def _ratio(top: float, bottom: float) -> float:
    """NaN when either side is missing or the bottom is zero."""
    if math.isnan(top) or math.isnan(bottom) or bottom == 0:
        return NAN
    return top / bottom


def _per_at_least_one(top: float, bottom: float) -> float:
    """top / max(bottom, 1), NaN when either side is missing."""
    return _ratio(top, max(bottom, 1) if not math.isnan(bottom) else NAN)


def _log1p(value: float) -> float:
    return NAN if math.isnan(value) else math.log1p(value)


def _age_days(now: datetime, timestamp: str | None) -> float:
    then = parse_time(timestamp)
    return NAN if then is None else (now - then).total_seconds() / 86400


def _tag_count(tags) -> float:
    if tags is None:
        return NAN
    try:
        return float(len(json.loads(tags) if isinstance(tags, str) else tags))
    except (TypeError, ValueError):
        return NAN


# ---------- metadata ----------

def metadata_features(row: dict, now: datetime) -> dict:
    """now is collected_at during training and the current time during serving.
    Missing values stay as NaN, LightGBM handles them."""
    title = row.get("title")
    desc = row.get("description")

    if title is None:
        title_features = dict.fromkeys(
            ["title_len", "title_word_count", "title_caps_ratio", "title_exclaim",
             "title_question", "title_digit_count", "title_emoji_count"], NAN
        )
    else:
        letters = [c for c in title if c.isalpha()]
        title_features = {
            "title_len": float(len(title)),
            "title_word_count": float(len(title.split())),
            "title_caps_ratio": _ratio(
                float(sum(c.isupper() for c in letters)), float(len(letters))
            ),
            "title_exclaim": float(title.count("!")),
            "title_question": float(title.count("?")),
            "title_digit_count": float(sum(c.isdigit() for c in title)),
            "title_emoji_count": float(len(EMOJI.findall(title))),
        }

    if desc is None:
        desc_features = dict.fromkeys(
            ["desc_len", "desc_link_count", "desc_hashtag_count", "desc_ai_terms"], NAN
        )
    else:
        desc_features = {
            "desc_len": float(len(desc)),
            "desc_link_count": float(desc.lower().count("http")),
            "desc_hashtag_count": float(len(HASHTAG.findall(desc))),
            "desc_ai_terms": float(len(AI_TERMS_PATTERN.findall(desc))),
        }

    duration = _number(row.get("duration_seconds"))
    views = _number(row.get("view_count"))
    video_age = _age_days(now, row.get("published_at"))
    channel_age = _age_days(now, row.get("channel_created_at"))
    channel_videos = _number(row.get("channel_video_count"))
    channel_subs = _number(row.get("channel_subs"))

    features = {
        **title_features,
        **desc_features,
        "tag_count": _tag_count(row.get("tags")),
        "duration_seconds": duration,
        "is_short": NAN if math.isnan(duration) else float(duration <= 180),
        "log_views": _log1p(views),
        "like_per_view": _ratio(_number(row.get("like_count")), views),
        "comment_per_view": _ratio(_number(row.get("comment_count")), views),
        "video_age_days": video_age,
        "views_per_day": _per_at_least_one(views, video_age),
        "disclosure_flag": _number(row.get("disclosure_flag")),
        "channel_age_days": channel_age,
        "channel_video_count": channel_videos,
        "uploads_per_day": _per_at_least_one(channel_videos, channel_age),
        "log_channel_subs": _log1p(channel_subs),
        "subs_per_video": _per_at_least_one(channel_subs, channel_videos),
    }
    return {name: features[name] for name in METADATA_FEATURES}


# ---------- visuals ----------

@lru_cache(maxsize=1)
def _load_clip():
    import open_clip
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL, pretrained=CLIP_PRETRAINED, device=device
    )
    model.eval()
    return model, preprocess, device


def image_embedding(paths: list[Path]) -> np.ndarray:
    """Encodes every image that exists, L2 normalizes each, averages, normalizes
    again. All NaN when there is no usable image."""
    import torch
    from PIL import Image, UnidentifiedImageError

    model, preprocess, device = _load_clip()

    images = []
    for path in paths:
        if path is None or not Path(path).is_file():
            continue
        try:
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        except (OSError, UnidentifiedImageError):
            continue
    if not images:
        return np.full(EMBEDDING_SIZE, np.nan, dtype=np.float32)

    with torch.no_grad():
        encoded = model.encode_image(torch.stack(images).to(device)).float()
    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    mean = encoded.mean(dim=0)
    return (mean / mean.norm()).cpu().numpy().astype(np.float32)


def build_visual_cache(cache_name: str = "visual",
                       image_columns: list[str] = IMAGE_COLUMNS) -> pd.DataFrame:
    """Writes data/features/<cache_name>.parquet for the train split. Only
    computes rows missing from the cache."""
    path = FEATURE_DIR / f"{cache_name}.parquet"
    cached = pd.read_parquet(path) if path.exists() else None
    done = set(cached["video_id"]) if cached is not None else set()

    new = []
    for row in load_train_rows():
        if row["video_id"] not in done:
            paths = [stored_path(row[c]) for c in image_columns]
            new.append([row["video_id"], *image_embedding(paths)])

    if new or cached is None:
        frame = pd.DataFrame(new, columns=["video_id"] + EMBEDDING_COLUMNS)
        if cached is not None:
            frame = pd.concat([cached, frame], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame
    return cached


# ---------- voice ----------

@lru_cache(maxsize=None)
def _load_voice_pipeline(model_id: str):
    import torch
    from transformers import pipeline

    return pipeline(
        "audio-classification", model=model_id,
        device=0 if torch.cuda.is_available() else -1,
    )


def voice_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Splits a clip into 5 second windows, dropping a very short tail."""
    size = VOICE_WINDOW_SECONDS * SAMPLE_RATE
    windows = [audio[i:i + size] for i in range(0, len(audio), size)]
    if len(windows) > 1 and len(windows[-1]) < MIN_WINDOW_SECONDS * SAMPLE_RATE:
        windows.pop()
    return windows


def voice_score(wav_path: Path, model_id: str, fake_label: str) -> float:
    """Mean probability of the fake label over 5 second windows. NaN if the file
    is missing or silent."""
    import soundfile as sf

    if wav_path is None or not Path(wav_path).is_file():
        return NAN
    audio, rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{wav_path} is {rate} Hz, expected {SAMPLE_RATE}")
    if audio.size == 0 or np.abs(audio).max() < SILENCE_PEAK:
        return NAN

    pipe = _load_voice_pipeline(model_id)
    scores = []
    for window in voice_windows(audio):
        results = pipe({"raw": window, "sampling_rate": SAMPLE_RATE}, top_k=None)
        by_label = {r["label"].lower(): r["score"] for r in results}
        scores.append(by_label[fake_label.lower()])
    return float(np.mean(scores))


def voice_cache_path(model_id: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id)
    return FEATURE_DIR / f"voice_{safe_name}.parquet"


def build_voice_cache(model_id: str, fake_label: str) -> pd.DataFrame:
    """Writes data/features/voice_<safe_model_name>.parquet for the train split.
    Only computes rows missing from the cache."""
    path = voice_cache_path(model_id)
    cached = pd.read_parquet(path) if path.exists() else None
    done = set(cached["video_id"]) if cached is not None else set()

    new = [
        {"video_id": row["video_id"],
         "voice": voice_score(stored_path(row["audio_path"]), model_id, fake_label)}
        for row in load_train_rows() if row["video_id"] not in done
    ]
    if new or cached is None:
        frame = pd.DataFrame(new, columns=["video_id", "voice"])
        if cached is not None:
            frame = pd.concat([cached, frame], ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        return frame
    return cached


if __name__ == "__main__":
    visual = build_visual_cache()
    print(f"visual cache {visual.shape}")
    frames_only = build_visual_cache("visual_frames", FRAME_COLUMNS)
    print(f"frames only visual cache {frames_only.shape}")
