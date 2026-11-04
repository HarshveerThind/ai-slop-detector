import pytest

from collect import extract_video_id, parse_duration

VIDEO_ID = "dQw4w9WgXcQ"


@pytest.mark.parametrize("url", [
    VIDEO_ID,
    f"  {VIDEO_ID}  ",
    f"https://www.youtube.com/watch?v={VIDEO_ID}",
    f"https://youtube.com/watch?v={VIDEO_ID}",
    f"https://m.youtube.com/watch?v={VIDEO_ID}",
    f"https://music.youtube.com/watch?v={VIDEO_ID}",
    f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLabc&index=2",
    f"https://www.youtube.com/watch?list=PLabc&v={VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}",
    f"https://youtu.be/{VIDEO_ID}?t=30",
    f"https://www.youtube.com/shorts/{VIDEO_ID}",
    f"https://www.youtube.com/embed/{VIDEO_ID}",
    f"https://www.youtube.com/live/{VIDEO_ID}",
])
def test_extract_video_id(url):
    assert extract_video_id(url) == VIDEO_ID


@pytest.mark.parametrize("url", [
    "",
    "not a link",
    "https://www.youtube.com/watch",
    "https://www.youtube.com/watch?v=short",
    "https://www.youtube.com/@somechannel",
    f"https://vimeo.com/{VIDEO_ID}",
])
def test_extract_video_id_rejects_bad_links(url):
    with pytest.raises(ValueError):
        extract_video_id(url)


@pytest.mark.parametrize("iso, seconds", [
    ("PT45S", 45),
    ("PT3M", 180),
    ("PT1H2M3S", 3723),
    ("PT1H", 3600),
    ("P1DT1S", 86401),
    ("P0D", 0),
    ("", 0),
    (None, 0),
    ("garbage", 0),
])
def test_parse_duration(iso, seconds):
    assert parse_duration(iso) == seconds
