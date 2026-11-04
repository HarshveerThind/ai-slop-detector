"""Scores one video. The API calls score_video, evaluate.py calls score_row
directly on stored data, so both go through exactly the same code."""

import json
import math
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from collect import fetch_for_scoring
from config import MODEL_DIR
from features import (
    IMAGE_COLUMNS, _load_clip, _load_voice_pipeline, image_embedding, metadata_features, stack_inputs, stored_path,
    voice_score,
)

REASON_CUTOFF = 0.7


@lru_cache(maxsize=1)
def load_models() -> dict:
    return {
        "config": json.loads((MODEL_DIR / "config.json").read_text()),
        "metadata": joblib.load(MODEL_DIR / "metadata_lgbm.joblib"),
        "visuals": joblib.load(MODEL_DIR / "visual_lr.joblib"),
        "stacker_full": joblib.load(MODEL_DIR / "stacker_full.joblib"),
        "stacker_no_voice": joblib.load(MODEL_DIR / "stacker_no_voice.joblib"),
    }


def warm_up() -> None:
    """Loads everything up front so the first request is not the slow one."""
    config = load_models()["config"]
    _load_clip()
    _load_voice_pipeline(config["voice_model_id"])


def model_version() -> str:
    return load_models()["config"]["model_version"]


def _rounded(score: float) -> float | None:
    return None if math.isnan(score) else round(score, 4)


def verdict_for(overall: float, unsure_band: list[float]) -> str:
    if overall > unsure_band[1]:
        return "likely_ai"
    if overall < unsure_band[0]:
        return "likely_human"
    return "unsure"


def reasons_for(voice: float, visuals: float, metadata: float, disclosed: bool) -> list[str]:
    """NaN compares False, so a missing sub-score never gives a reason."""
    reasons = []
    if voice >= REASON_CUTOFF:
        reasons.append("Synthetic sounding voice")
    if visuals >= REASON_CUTOFF:
        reasons.append("AI generated visuals")
    if disclosed:
        reasons.append("Creator disclosed AI content")
    if metadata >= REASON_CUTOFF:
        reasons.append("Upload patterns common in AI channels")
    return reasons


def score_row(row: dict, image_paths: list[Path], wav_path: Path | None, now: datetime) -> dict:
    """The serving path. Turns raw metadata, images and audio into the result."""
    models = load_models()
    config = models["config"]

    features = metadata_features(row, now)
    metadata_X = pd.DataFrame([features])[config["METADATA_FEATURES"]]
    metadata = float(models["metadata"].predict_proba(metadata_X)[0, 1])

    embedding = image_embedding(image_paths)
    if pd.isna(embedding).any():
        visuals = math.nan
    else:
        visual_X = pd.DataFrame([embedding], columns=models["visuals"].feature_names_in_)
        visuals = float(models["visuals"].predict_proba(visual_X)[0, 1])

    voice = math.nan
    if wav_path is not None:
        voice = voice_score(wav_path, config["voice_model_id"], config["voice_fake_label"])

    audio_used = not math.isnan(voice)
    stacker = models["stacker_full"] if audio_used else models["stacker_no_voice"]
    columns = config["stack_full"] if audio_used else config["stack_no_voice"]
    scores = {"metadata": metadata, "visuals": visuals, "voice": voice}
    stack_X = stack_inputs(pd.DataFrame([{
        c: config["fill_values"][c] if math.isnan(scores[c]) else scores[c] for c in columns
    }])[columns])
    overall = float(stacker.predict_proba(stack_X)[0, 1])

    return {
        "video_id": row["video_id"],
        "overall": round(overall, 4),
        "verdict": verdict_for(overall, config["unsure_band"]),
        "voice": _rounded(voice),
        "visuals": _rounded(visuals),
        "metadata": _rounded(metadata),
        "reasons": reasons_for(voice, visuals, metadata, row.get("disclosure_flag") == 1),
        "audio_used": audio_used,
        "model_version": config["model_version"],
    }


def score_video(video_id: str, audio_path: Path | None) -> dict:
    """Fetches what is needed for one live video and scores it."""
    with tempfile.TemporaryDirectory() as tmp:
        row = fetch_for_scoring(video_id, Path(tmp))
        image_paths = [stored_path(row[c]) for c in IMAGE_COLUMNS]
        return score_row(row, image_paths, audio_path, datetime.now(timezone.utc))
