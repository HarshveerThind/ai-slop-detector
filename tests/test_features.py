import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from config import SAMPLE_RATE
from features import (
    METADATA_FEATURES, metadata_features, stack_inputs, stored_path, voice_score,
    voice_windows,
)

NOW = datetime(2026, 1, 11, tzinfo=timezone.utc)

ROW = {
    "title": "WOW 2 Cats! Why? \U0001F600",
    "description": "Made with AI-generated art and ElevenLabs tts. "
                   "See http://a.com https://b.com #cats #sora veoh",
    "tags": '["a", "b", "c"]',
    "duration_seconds": 120,
    "view_count": 999,
    "like_count": 100,
    "comment_count": 50,
    "published_at": "2026-01-01T00:00:00Z",
    "disclosure_flag": 1,
    "channel_created_at": "2025-12-12T00:00:00Z",
    "channel_video_count": 60,
    "channel_subs": 299,
}


@pytest.mark.parametrize("name, expected", [
    ("title_len", 18),
    ("title_word_count", 5),
    ("title_caps_ratio", 5 / 10),
    ("title_exclaim", 1),
    ("title_question", 1),
    ("title_digit_count", 1),
    ("title_emoji_count", 1),
    ("desc_len", len(ROW["description"])),
    ("desc_link_count", 2),
    ("desc_hashtag_count", 2),
    ("desc_ai_terms", 4),     # ai-generated, elevenlabs, tts, sora. not veoh
    ("tag_count", 3),
    ("duration_seconds", 120),
    ("is_short", 1),
    ("log_views", math.log1p(999)),
    ("like_per_view", 100 / 999),
    ("comment_per_view", 50 / 999),
    ("video_age_days", 10),
    ("views_per_day", 99.9),
    ("disclosure_flag", 1),
    ("channel_age_days", 30),
    ("channel_video_count", 60),
    ("uploads_per_day", 2),
    ("log_channel_subs", math.log1p(299)),
    ("subs_per_video", 299 / 60),
])
def test_every_feature_on_a_hand_built_row(name, expected):
    assert metadata_features(ROW, NOW)[name] == pytest.approx(expected)


def test_every_listed_feature_is_covered_and_in_order():
    assert list(metadata_features(ROW, NOW)) == METADATA_FEATURES


def test_empty_row_is_all_nan():
    features = metadata_features({}, NOW)
    assert all(math.isnan(v) for v in features.values())


@pytest.mark.parametrize("missing, nan_features", [
    ("like_count", ["like_per_view"]),
    ("comment_count", ["comment_per_view"]),
    ("channel_subs", ["log_channel_subs", "subs_per_video"]),
    ("view_count", ["log_views", "like_per_view", "comment_per_view", "views_per_day"]),
    ("published_at", ["video_age_days", "views_per_day"]),
    ("channel_created_at", ["channel_age_days", "uploads_per_day"]),
    ("channel_video_count", ["channel_video_count", "uploads_per_day", "subs_per_video"]),
    ("duration_seconds", ["duration_seconds", "is_short"]),
    ("tags", ["tag_count"]),
    ("disclosure_flag", ["disclosure_flag"]),
])
def test_missing_values_only_touch_their_features(missing, nan_features):
    features = metadata_features({**ROW, missing: None}, NOW)
    assert sorted(k for k, v in features.items() if math.isnan(v)) == sorted(nan_features)


def test_edge_values():
    row = {**ROW, "title": "123 !!", "description": "", "tags": "[]",
           "view_count": 0, "duration_seconds": 181,
           "published_at": "2026-01-11T00:00:00Z", "channel_video_count": 0}
    features = metadata_features(row, NOW)
    assert math.isnan(features["title_caps_ratio"])    # no letters
    assert features["desc_len"] == 0
    assert features["tag_count"] == 0
    assert math.isnan(features["like_per_view"])       # zero views
    assert features["is_short"] == 0
    assert features["views_per_day"] == 0              # age floored at 1 day
    assert features["subs_per_video"] == 299           # count floored at 1


def test_is_short_boundary():
    assert metadata_features({**ROW, "duration_seconds": 180}, NOW)["is_short"] == 1


def test_stored_path_handles_windows_separators():
    path = stored_path("data\\raw\\audio\\x.wav")
    assert path.parts[-4:] == ("data", "raw", "audio", "x.wav")
    assert path.is_absolute()
    assert stored_path(None) is None
    assert stored_path("") is None


def test_stack_inputs_are_logits_and_stay_finite():
    out = stack_inputs(pd.DataFrame({"a": [0.5, 0.0, 1.0, 0.75]}))["a"].tolist()
    assert out[0] == pytest.approx(0)
    assert out[1] == pytest.approx(-out[2])
    assert all(math.isfinite(v) for v in out)
    assert out[3] == pytest.approx(math.log(3))


def test_voice_windows_cut_into_five_seconds_and_drop_tiny_tail():
    audio = np.zeros(SAMPLE_RATE * 20)
    assert [len(w) for w in voice_windows(audio)] == [SAMPLE_RATE * 5] * 4
    audio = np.zeros(SAMPLE_RATE * 10 + 100)
    assert len(voice_windows(audio)) == 2
    audio = np.zeros(SAMPLE_RATE * 12)
    assert [len(w) for w in voice_windows(audio)][-1] == SAMPLE_RATE * 2
    audio = np.zeros(100)
    assert len(voice_windows(audio)) == 1      # never drop the only window


def test_voice_score_missing_or_silent_is_nan(tmp_path):
    import soundfile as sf

    assert math.isnan(voice_score(tmp_path / "nope.wav", "unused", "fake"))
    assert math.isnan(voice_score(None, "unused", "fake"))
    silent = tmp_path / "silent.wav"
    sf.write(silent, np.zeros(SAMPLE_RATE), SAMPLE_RATE)
    assert math.isnan(voice_score(silent, "unused", "fake"))
