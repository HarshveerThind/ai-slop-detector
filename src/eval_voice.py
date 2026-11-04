"""Compares the pretrained voice detectors on the train split.

Run from the repo root:
    python src/eval_voice.py

Writes reports/voice_eval.json. Set VOICE_MODEL in config.py to the chosen one.
"""

import json
import time
from datetime import datetime, timezone

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from config import REPORT_DIR, VOICE_CANDIDATES
from features import build_voice_cache, load_train_rows


def evaluate_candidate(rows: pd.DataFrame, model_id: str, fake_label: str) -> dict:
    start = time.perf_counter()
    scores = build_voice_cache(model_id, fake_label)
    seconds = time.perf_counter() - start

    joined = rows.merge(scores, on="video_id", how="inner")
    usable = joined.dropna(subset=["voice"])
    per_channel = (
        usable.groupby(["channel_title", "ai_voice"])["voice"].mean().round(3)
        .reset_index().sort_values("voice", ascending=False)
    )
    return {
        "model_id": model_id,
        "fake_label": fake_label,
        "clips_scored": len(usable),
        "clips_no_score": len(joined) - len(usable),
        "roc_auc": roc_auc_score(usable["ai_voice"], usable["voice"]),
        "average_precision": average_precision_score(usable["ai_voice"], usable["voice"]),
        "mean_score_ai_voice": float(usable.loc[usable["ai_voice"] == 1, "voice"].mean()),
        "mean_score_not_ai_voice": float(usable.loc[usable["ai_voice"] == 0, "voice"].mean()),
        "seconds_to_build_cache": round(seconds, 1),
        "per_channel": per_channel.to_dict(orient="records"),
    }


def main() -> None:
    rows = pd.DataFrame(load_train_rows())[["video_id", "channel_title", "ai_voice"]]
    print(f"{len(rows)} train clips, {int(rows['ai_voice'].sum())} with an AI voice")

    results = []
    for candidate in VOICE_CANDIDATES:
        print(f"Scoring with {candidate['model_id']}")
        try:
            results.append(evaluate_candidate(rows, **candidate))
        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({**candidate, "error": str(e)})
    results.sort(key=lambda r: r.get("roc_auc", -1), reverse=True)

    print(f"\n{'model':<62}{'ROC AUC':>9}{'AP':>7}{'ai mean':>9}{'real mean':>11}")
    for r in results:
        if "error" in r:
            print(f"{r['model_id']:<62}  failed")
        else:
            print(f"{r['model_id']:<62}{r['roc_auc']:>9.3f}{r['average_precision']:>7.3f}"
                  f"{r['mean_score_ai_voice']:>9.3f}{r['mean_score_not_ai_voice']:>11.3f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "voice_eval.json"
    out.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "train_clips": len(rows),
        "positives": int(rows["ai_voice"].sum()),
        "results": results,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
