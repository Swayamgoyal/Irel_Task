"""
Configuration for the Whisper Large pipeline.

All outputs are written to  data_whisper/<video_id>/  so they can be
compared side-by-side with the Sarvam pipeline outputs in  data/<video_id>/.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_whisper"          # separate from Sarvam's data/

# Legacy paths kept for reference
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR  = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUTS_DIR = DATA_DIR / "outputs"


def get_video_dir(video_id: str) -> Path:
    """Get or create a per-video data directory under data_whisper/."""
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d


# GitHub Models LLM config (shared with Sarvam pipeline)
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
LLM_MODEL_LIGHT = os.getenv("LLM_MODEL_LIGHT", "gpt-4o-mini")
LLM_MODEL       = os.getenv("LLM_MODEL",       "Phi-4")
LLM_MODEL_REASON= os.getenv("LLM_MODEL_REASON", "Phi-4")
LLM_BASE_URL    = "https://models.inference.ai.azure.com"

# Whisper model — force large-v3 for maximum accuracy
WHISPER_MODEL  = os.getenv("WHISPER_LARGE_MODEL", "large-v3")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")

# Audio config
AUDIO_SAMPLE_RATE = 16000
AUDIO_FORMAT      = "wav"
