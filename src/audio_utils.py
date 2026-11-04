"""Audio conversion shared by the collector and the API, so training audio
and live audio go through exactly the same step."""

import subprocess
from pathlib import Path

from config import SAMPLE_RATE


def to_wav16k(src_path: Path, dst_path: Path, max_seconds: int | None = None) -> None:
    """Converts any audio file to 16 kHz mono WAV. Raises if ffmpeg fails."""
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    limit = ["-t", str(max_seconds)] if max_seconds else []
    ff = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src_path),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            *limit,
            str(dst_path),
        ],
        capture_output=True, text=True,
    )
    if ff.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src_path}: {ff.stderr.strip()}")
