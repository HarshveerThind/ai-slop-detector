"""One time evaluation on the test split, through the real serving path.

Run from the repo root:
    python src/evaluate.py

This is the only file that reads the test split. Do not tune anything based on
what it prints.
"""

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np

from config import REPORT_DIR
from features import IMAGE_COLUMNS, parse_time, stored_path
from init_db import DB_PATH
from predict import load_models, score_row
from train import binary_metrics, print_metrics_table


def load_test_rows() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM videos WHERE split = 'test' AND label IS NOT NULL "
            "ORDER BY video_id"
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


def as_scores(values: list) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def main() -> None:
    rows = load_test_rows()
    config = load_models()["config"]
    threshold, sub = config["threshold"], config["sub_thresholds"]   # all fixed during training

    results = []
    for row in rows:
        image_paths = [stored_path(row[c]) for c in IMAGE_COLUMNS]
        wav_path = stored_path(row["audio_path"])
        results.append(score_row(row, image_paths, wav_path, parse_time(row["collected_at"])))

    label = [r["label"] for r in rows]
    overall = as_scores([r["overall"] for r in results])
    voice = as_scores([r["voice"] for r in results])
    visuals = as_scores([r["visuals"] for r in results])
    metadata = as_scores([r["metadata"] for r in results])

    # no-voice path: same rows with no audio, what happens when capture fails
    no_voice_results = []
    for row in rows:
        image_paths = [stored_path(row[c]) for c in IMAGE_COLUMNS]
        no_voice_results.append(score_row(row, image_paths, None, parse_time(row["collected_at"])))
    no_voice = as_scores([r["overall"] for r in no_voice_results])

    metrics = {
        "stacker_full vs label": binary_metrics(label, overall, threshold),
        "stacker_no_voice vs label": binary_metrics(label, no_voice, threshold),
        "metadata vs label": binary_metrics(label, metadata, sub["metadata"]),
        "visuals vs ai_visuals": binary_metrics([r["ai_visuals"] for r in rows], visuals, sub["visuals"]),
        "voice vs ai_voice": binary_metrics([r["ai_voice"] for r in rows], voice, sub["voice"]),
    }

    print(f"{len(rows)} test rows from {len({r['channel_id'] for r in rows})} channels")
    print_metrics_table(metrics)

    per_video = [
        {"video_id": row["video_id"], "channel_title": row["channel_title"],
         "label": row["label"], "ai_voice": row["ai_voice"], "ai_visuals": row["ai_visuals"],
         **{k: result[k] for k in ("overall", "verdict", "voice", "visuals", "metadata")}}
        for row, result in zip(rows, results)
    ]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "test_metrics.json"
    out.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_version": config["model_version"],
        "test_rows": len(rows),
        "test_channels": len({r["channel_id"] for r in rows}),
        "metrics": metrics,
        "per_video": per_video,
    }, indent=2))
    print(f"\nWrote {out}")
    print("\nWARNING: this test set should not be used for further tuning.")


if __name__ == "__main__":
    main()
