"""Constants shared by the whole project."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURE_DIR = DATA_DIR / "features"
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
REPORT_DIR = ROOT / "reports"

SAMPLE_RATE = 16000

CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

TEST_FRACTION = 0.20
RANDOM_SEED = 42

# counted case-insensitively in the description, whole words only
AI_TERMS = [
    "ai generated", "ai voice", "text to speech", "tts", "midjourney", "sora",
    "veo", "kling", "runway", "elevenlabs", "stable diffusion", "synthetic",
]

# pretrained synthetic speech detectors compared by eval_voice.py
VOICE_CANDIDATES = [
    {"model_id": "garystafford/wav2vec2-deepfake-voice-detector", "fake_label": "fake"},
    {"model_id": "Gustking/wav2vec2-large-xlsr-deepfake-audio-classification", "fake_label": "fake"},
    {"model_id": "MelodyMachine/Deepfake-audio-detection-V2", "fake_label": "fake"},
    {"model_id": "abhishtagatya/wav2vec2-base-960h-itw-deepfake", "fake_label": "spoof"},
]
