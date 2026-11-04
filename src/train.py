"""Trains the sub-models and the stackers on the train split only.

Run from the repo root:
    python src/train.py

Writes the artifacts to models/ and the metrics to reports/cv_metrics.json.
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from config import (
    CLIP_MODEL, CLIP_PRETRAINED, MODEL_DIR, RANDOM_SEED, REPORT_DIR, VOICE_MODEL,
)
from features import (
    EMBEDDING_COLUMNS, FRAME_COLUMNS, METADATA_FEATURES, build_visual_cache,
    build_voice_cache, load_train_rows, metadata_features, parse_time, stack_inputs,
)

N_FOLDS = 5
VISUAL_C_VALUES = [0.1, 1, 10]
UNSURE_MARGIN = 0.15
STACK_FULL = ["metadata", "visuals", "voice"]
STACK_NO_VOICE = ["metadata", "visuals"]


def new_metadata_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        class_weight="balanced", random_state=RANDOM_SEED, verbose=-1,
    )


def new_visual_model(c: float) -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000, C=c)


# ---------- metrics, also used by evaluate.py ----------

def best_f1_threshold(y, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def binary_metrics(y, scores, threshold: float) -> dict:
    """Rows with no score are left out. AUC and AP are null when the rows only
    have one class, since they mean nothing there."""
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    keep = ~np.isnan(scores) & ~np.isnan(y)
    y, scores = y[keep].astype(int), scores[keep]
    two_classes = len(set(y)) == 2
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "threshold": round(float(threshold), 4),
        "roc_auc": float(roc_auc_score(y, scores)) if two_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if two_classes else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def own_threshold_metrics(y, scores) -> dict:
    """Metrics at the score's own best F1 threshold, since sub-scores are not on
    the same scale as the stacker."""
    y, scores = np.asarray(y, dtype=float), np.asarray(scores, dtype=float)
    keep = ~np.isnan(scores)
    return binary_metrics(y, scores, best_f1_threshold(y[keep], scores[keep]))


def print_metrics_table(metrics: dict) -> None:
    print(f"{'':<34}{'rows':>5}{'pos':>5}{'AUC':>7}{'AP':>7}{'thr':>7}"
          f"{'prec':>7}{'rec':>7}{'F1':>7}   tn/fp/fn/tp")
    for name, m in metrics.items():
        auc = f"{m['roc_auc']:.3f}" if m["roc_auc"] is not None else "n/a"
        ap = f"{m['average_precision']:.3f}" if m["average_precision"] is not None else "n/a"
        cm = m["confusion_matrix"]
        print(f"{name:<34}{m['rows']:>5}{m['positives']:>5}{auc:>7}{ap:>7}"
              f"{m['threshold']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}{m['f1']:>7.3f}"
              f"   {cm['tn']}/{cm['fp']}/{cm['fn']}/{cm['tp']}")


# ---------- out of fold scores ----------

def out_of_fold(new_model, X: pd.DataFrame, y: np.ndarray, folds: list, usable=None) -> np.ndarray:
    """Probability for each row from a model that never saw that row's channel.
    Rows that are not usable are not trained on and stay NaN."""
    usable = np.ones(len(X), dtype=bool) if usable is None else usable
    scores = np.full(len(X), np.nan)
    for fit_idx, score_idx in folds:
        fit_idx, score_idx = fit_idx[usable[fit_idx]], score_idx[usable[score_idx]]
        model = new_model().fit(X.iloc[fit_idx], y[fit_idx])
        scores[score_idx] = model.predict_proba(X.iloc[score_idx])[:, 1]
    return scores


def select_visual_c(X: pd.DataFrame, y: np.ndarray, folds: list, usable: np.ndarray):
    """Picks C by grouped CV ROC AUC. Returns (best C, its OOF scores, AUC per C)."""
    best_c, best_scores, by_c = None, None, {}
    for c in VISUAL_C_VALUES:
        scores = out_of_fold(lambda: new_visual_model(c), X, y, folds, usable)
        by_c[c] = float(roc_auc_score(y[usable], scores[usable]))
        if best_c is None or by_c[c] > by_c[best_c]:
            best_c, best_scores = c, scores
    return best_c, best_scores, by_c


def load_training_frame() -> pd.DataFrame:
    rows = load_train_rows()
    features = pd.DataFrame(
        [metadata_features(r, parse_time(r["collected_at"])) for r in rows]
    )[METADATA_FEATURES]
    base = pd.DataFrame(rows)[
        ["video_id", "channel_id", "channel_title", "label",
         "ai_voice", "ai_visuals", "ai_thumbnail", "disclosure_flag"]
    ].rename(columns={"disclosure_flag": "raw_disclosure"})
    frame = pd.concat([base, features], axis=1)
    frame = frame.merge(build_visual_cache(), on="video_id", how="left")
    voice = build_voice_cache(VOICE_MODEL["model_id"], VOICE_MODEL["fake_label"])
    return frame.merge(voice, on="video_id", how="left")


def main() -> None:
    frame = load_training_frame()
    labels = frame["label"].to_numpy()
    visual_labels = frame["ai_visuals"].to_numpy()
    print(f"{len(frame)} train rows from {frame['channel_id'].nunique()} channels, "
          f"{int(labels.sum())} positive")

    folds = list(GroupKFold(n_splits=N_FOLDS).split(frame, labels, frame["channel_id"]))

    # sub-models, out of fold
    metadata_X = frame[METADATA_FEATURES]
    metadata_oof = out_of_fold(new_metadata_model, metadata_X, labels, folds)

    visual_X = frame[EMBEDDING_COLUMNS]
    has_embedding = visual_X.notna().all(axis=1).to_numpy()
    best_c, visual_oof, auc_by_c = select_visual_c(visual_X, visual_labels, folds, has_embedding)
    print(f"Visuals C by grouped CV AUC {auc_by_c}, using C={best_c}")

    voice = frame["voice"].to_numpy(dtype=float)

    # thumbnail comparison, same folds and same C search but no thumbnail in the embedding
    frames_only = frame[["video_id"]].merge(
        build_visual_cache("visual_frames", FRAME_COLUMNS), on="video_id", how="left"
    )[EMBEDDING_COLUMNS]
    frames_usable = frames_only.notna().all(axis=1).to_numpy()
    frames_c, frames_oof, _ = select_visual_c(frames_only, visual_labels, folds, frames_usable)

    # stackers. NaN sub-scores become the train mean, and those means are saved
    scores = pd.DataFrame({"metadata": metadata_oof, "visuals": visual_oof, "voice": voice})
    fill_values = {name: float(scores[name].mean()) for name in STACK_FULL}
    filled = stack_inputs(scores.fillna(fill_values))

    new_stacker = lambda: LogisticRegression(max_iter=2000)
    full_oof = out_of_fold(new_stacker, filled[STACK_FULL], labels, folds)
    no_voice_oof = out_of_fold(new_stacker, filled[STACK_NO_VOICE], labels, folds)

    threshold = best_f1_threshold(labels, full_oof)
    unsure_band = [max(threshold - UNSURE_MARGIN, 0.0), min(threshold + UNSURE_MARGIN, 1.0)]

    # refit on all train rows
    metadata_model = new_metadata_model().fit(metadata_X, labels)
    visual_model = new_visual_model(best_c).fit(visual_X[has_embedding], visual_labels[has_embedding])
    stacker_full = new_stacker().fit(filled[STACK_FULL], labels)
    stacker_no_voice = new_stacker().fit(filled[STACK_NO_VOICE], labels)

    metrics = {
        "stacker_full vs label": binary_metrics(labels, full_oof, threshold),
        "stacker_no_voice vs label": binary_metrics(labels, no_voice_oof, threshold),
        "metadata vs label": own_threshold_metrics(labels, metadata_oof),
        "visuals vs ai_visuals": own_threshold_metrics(visual_labels, visual_oof),
        "voice vs ai_voice": own_threshold_metrics(frame["ai_voice"], voice),
    }
    baselines = {
        "disclosure flag vs label": binary_metrics(labels, frame["raw_disclosure"], 0.5),
        "metadata alone vs label": own_threshold_metrics(labels, metadata_oof),
        "visuals alone vs label": own_threshold_metrics(labels, visual_oof),
        "voice alone vs label": own_threshold_metrics(labels, voice),
    }

    # did the thumbnail mislead the visuals model? the group that matters is
    # AI thumbnail on a video whose visuals are not AI
    thumbnail_only = ((frame["ai_thumbnail"] == 1) & (frame["ai_visuals"] == 0)).to_numpy()
    fully_real = (frame["label"] == 0).to_numpy()
    visual_comparison = {
        "thumbnail_plus_frames": own_threshold_metrics(visual_labels, visual_oof),
        "frames_only": own_threshold_metrics(visual_labels, frames_oof),
        "frames_only_best_c": frames_c,
        "ai_thumbnail_but_real_visuals_rows": int(thumbnail_only.sum()),
        "mean_score_ai_thumbnail_but_real_visuals": {
            "thumbnail_plus_frames": float(np.nanmean(visual_oof[thumbnail_only])),
            "frames_only": float(np.nanmean(frames_oof[thumbnail_only])),
        },
        "mean_score_fully_real": {
            "thumbnail_plus_frames": float(np.nanmean(visual_oof[fully_real])),
            "frames_only": float(np.nanmean(frames_oof[fully_real])),
        },
    }

    gains = metadata_model.booster_.feature_importance(importance_type="gain")
    importances = sorted(zip(METADATA_FEATURES, map(float, gains)), key=lambda p: -p[1])

    # stacker weights are worth seeing, they say what the overall score leans on
    stacker_weights = {
        "stacker_full": dict(zip(STACK_FULL, map(float, stacker_full.coef_[0]))),
        "stacker_no_voice": dict(zip(STACK_NO_VOICE, map(float, stacker_no_voice.coef_[0]))),
    }

    # save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(metadata_model, MODEL_DIR / "metadata_lgbm.joblib")
    joblib.dump(visual_model, MODEL_DIR / "visual_lr.joblib")
    joblib.dump(stacker_full, MODEL_DIR / "stacker_full.joblib")
    joblib.dump(stacker_no_voice, MODEL_DIR / "stacker_no_voice.joblib")

    model_version = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (MODEL_DIR / "config.json").write_text(json.dumps({
        "model_version": model_version,
        "METADATA_FEATURES": METADATA_FEATURES,
        "clip_model": CLIP_MODEL,
        "clip_pretrained": CLIP_PRETRAINED,
        "voice_model_id": VOICE_MODEL["model_id"],
        "voice_fake_label": VOICE_MODEL["fake_label"],
        "stack_full": STACK_FULL,
        "stack_no_voice": STACK_NO_VOICE,
        "fill_values": fill_values,
        "threshold": threshold,
        "unsure_band": unsure_band,
        "sub_thresholds": {
            "metadata": metrics["metadata vs label"]["threshold"],
            "visuals": metrics["visuals vs ai_visuals"]["threshold"],
            "voice": metrics["voice vs ai_voice"]["threshold"],
        },
        "visual_c": best_c,
        "training_rows": {
            "total": len(frame),
            "channels": int(frame["channel_id"].nunique()),
            "label_positive": int(labels.sum()),
            "ai_visuals_positive": int(visual_labels.sum()),
            "ai_voice_positive": int(frame["ai_voice"].sum()),
            "with_embedding": int(has_embedding.sum()),
            "with_voice": int(np.sum(~np.isnan(voice))),
        },
    }, indent=2))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "cv_metrics.json").write_text(json.dumps({
        "model_version": model_version,
        "threshold": threshold,
        "unsure_band": unsure_band,
        "visual_auc_by_c": {str(k): v for k, v in auc_by_c.items()},
        "metrics": metrics,
        "baselines": baselines,
        "visual_comparison": visual_comparison,
        "stacker_weights": stacker_weights,
        "feature_importance_gain": dict(importances),
    }, indent=2))

    print(f"\nThreshold {threshold:.3f}, unsure band {unsure_band[0]:.3f} to {unsure_band[1]:.3f}")
    print("\nCross validated metrics")
    print_metrics_table(metrics)
    print("\nBaselines against label")
    print_metrics_table(baselines)
    print("\nVisuals, thumbnail plus frames vs frames only")
    print_metrics_table({k: visual_comparison[k] for k in ("thumbnail_plus_frames", "frames_only")})
    print(f"  mean score on AI thumbnail but real visuals: "
          f"{visual_comparison['mean_score_ai_thumbnail_but_real_visuals']}")
    print(f"  mean score on fully real videos: {visual_comparison['mean_score_fully_real']}")
    print(f"\nStacker weights {stacker_weights}")
    print("\nTop 10 metadata features by gain")
    for name, gain in importances[:10]:
        print(f"  {name:<24}{gain:>10.1f}")
    print(f"\nSaved models to {MODEL_DIR}, version {model_version}")


if __name__ == "__main__":
    main()
