import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import api

VIDEO_ID = "dQw4w9WgXcQ"
RESULT = {
    "video_id": VIDEO_ID, "overall": 0.87, "verdict": "likely_ai", "voice": 0.94,
    "visuals": 0.71, "metadata": 0.4, "reasons": ["Synthetic sounding voice"],
    "audio_used": True, "model_version": "v1",
}


@pytest.fixture
def scoring(monkeypatch, tmp_path):
    """Temp cache, fake models. Returns the list of (video_id, wav existed) calls."""
    calls = []

    def fake_score(video_id, audio_path):
        calls.append((video_id, audio_path is not None and audio_path.exists()))
        return {**RESULT, "video_id": video_id}

    def fake_convert(src, dst):
        dst.write_bytes(b"wav")

    monkeypatch.setattr(api, "CACHE_DB", tmp_path / "cache.db")
    monkeypatch.setattr(api, "score_video", fake_score)
    monkeypatch.setattr(api, "to_wav16k", fake_convert)
    monkeypatch.setattr(api, "model_version", lambda: "v1")
    api.request_times.clear()
    return calls


# no "with", so the lifespan handler does not load the real models
client = TestClient(api.app)


def age_cache_row(video_id, **days):
    conn = sqlite3.connect(api.CACHE_DB)
    conn.execute(
        "UPDATE scores SET created_at = ? WHERE video_id = ?",
        ((datetime.now(timezone.utc) - timedelta(**days)).isoformat(), video_id),
    )
    conn.commit()
    conn.close()


def test_health(scoring):
    assert client.get("/health").json() == {"status": "ok", "model_version": "v1"}


@pytest.mark.parametrize("bad", ["short", "waytoolongvideoid", "has spaces!", "abc$defghij"])
def test_bad_video_ids_rejected(scoring, bad):
    assert client.post("/score", data={"video_id": bad}).status_code == 400
    assert client.get(f"/score/{bad}").status_code == 400
    assert scoring == []


def test_missing_video_id_rejected(scoring):
    assert client.post("/score", data={}).status_code == 422


def test_oversized_audio_rejected(scoring):
    big = b"x" * (api.MAX_AUDIO_BYTES + 1)
    response = client.post("/score", data={"video_id": VIDEO_ID},
                           files={"audio": ("clip.webm", big, "audio/webm")})
    assert response.status_code == 413
    assert scoring == []


def test_audio_at_the_limit_accepted(scoring):
    exact = b"x" * api.MAX_AUDIO_BYTES
    response = client.post("/score", data={"video_id": VIDEO_ID},
                           files={"audio": ("clip.webm", exact, "audio/webm")})
    assert response.status_code == 200


def test_cache_miss_is_404(scoring):
    assert client.get(f"/score/{VIDEO_ID}").status_code == 404


def test_score_with_audio_then_cache_hit(scoring):
    response = client.post("/score", data={"video_id": VIDEO_ID},
                           files={"audio": ("clip.webm", b"audio", "audio/webm")})
    assert response.status_code == 200
    assert response.json()["overall"] == 0.87
    assert scoring == [(VIDEO_ID, True)]

    hit = client.get(f"/score/{VIDEO_ID}")
    assert hit.status_code == 200
    assert hit.json() == response.json()
    assert len(scoring) == 1       # served from the cache, not scored again


def test_score_with_missing_audio(scoring):
    response = client.post("/score", data={"video_id": VIDEO_ID})
    assert response.status_code == 200
    assert scoring == [(VIDEO_ID, False)]


def test_post_always_rescores(scoring):
    client.post("/score", data={"video_id": VIDEO_ID})
    client.post("/score", data={"video_id": VIDEO_ID})
    assert len(scoring) == 2


def test_cache_ignored_for_other_model_version(scoring, monkeypatch):
    client.post("/score", data={"video_id": VIDEO_ID})
    monkeypatch.setattr(api, "model_version", lambda: "v2")
    assert client.get(f"/score/{VIDEO_ID}").status_code == 404


def test_cache_expires_after_seven_days(scoring):
    client.post("/score", data={"video_id": VIDEO_ID})
    age_cache_row(VIDEO_ID, days=6, hours=23)
    assert client.get(f"/score/{VIDEO_ID}").status_code == 200
    age_cache_row(VIDEO_ID, days=7)
    assert client.get(f"/score/{VIDEO_ID}").status_code == 404


def test_unreadable_audio_is_400_and_not_cached(scoring, monkeypatch):
    def broken(src, dst):
        raise RuntimeError("ffmpeg failed")
    monkeypatch.setattr(api, "to_wav16k", broken)
    response = client.post("/score", data={"video_id": VIDEO_ID},
                           files={"audio": ("clip.webm", b"junk", "audio/webm")})
    assert response.status_code == 400
    assert client.get(f"/score/{VIDEO_ID}").status_code == 404


def test_missing_video_is_404_and_other_failures_502(scoring, monkeypatch):
    def not_found(video_id, audio_path):
        raise ValueError("Video not found or not public")
    monkeypatch.setattr(api, "score_video", not_found)
    assert client.post("/score", data={"video_id": VIDEO_ID}).status_code == 404

    def quota_gone(video_id, audio_path):
        raise RuntimeError("secret detail")
    monkeypatch.setattr(api, "score_video", quota_gone)
    response = client.post("/score", data={"video_id": VIDEO_ID})
    assert response.status_code == 502
    assert "secret" not in response.text


def test_rate_limit_is_per_ip_and_skips_health(scoring):
    first_ip = {"x-forwarded-for": "1.1.1.1"}
    for _ in range(api.RATE_LIMIT):
        assert client.get(f"/score/{VIDEO_ID}", headers=first_ip).status_code == 404
    assert client.get(f"/score/{VIDEO_ID}", headers=first_ip).status_code == 429
    assert client.get("/health", headers=first_ip).status_code == 200
    assert client.get(f"/score/{VIDEO_ID}",
                      headers={"x-forwarded-for": "2.2.2.2"}).status_code == 404


def test_rate_limit_uses_the_address_the_load_balancer_added(scoring):
    for i in range(api.RATE_LIMIT):
        client.get(f"/score/{VIDEO_ID}", headers={"x-forwarded-for": f"9.9.9.{i}, 1.1.1.1"})
    spoofed = {"x-forwarded-for": "8.8.8.8, 1.1.1.1"}
    assert client.get(f"/score/{VIDEO_ID}", headers=spoofed).status_code == 429
