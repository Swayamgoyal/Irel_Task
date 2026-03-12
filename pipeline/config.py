"""
Configuration for the pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Legacy paths (kept for backward compat, new pipeline uses per-video dirs)
VIDEOS_DIR = DATA_DIR / "videos"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
OUTPUTS_DIR = DATA_DIR / "outputs"


def get_video_dir(video_id: str) -> Path:
    """Get or create a per-video data directory."""
    d = DATA_DIR / video_id
    d.mkdir(parents=True, exist_ok=True)
    return d

# GitHub Models LLM config (OpenAI-compatible)
# Endpoint: https://models.inference.ai.azure.com
#
# Model tiers — tuned per task:
#   LLM_MODEL_LIGHT   : fast/cheap — ASR spelling correction (many small calls)
#   LLM_MODEL         : standard   — normalization (understands Indian languages well)
#   LLM_MODEL_REASON  : best       — concept extraction + prerequisite mapping
#
# GitHub Models free tier supports:
#   gpt-4o-mini           — fast, decent multilingual
#   gpt-4.1-mini          — newer, better reasoning than 4o-mini
#   Phi-4                 — Microsoft, excellent multilingual/Indian lang, generous quota
#   Meta-Llama-3.3-70B-Instruct — very capable, good for structured JSON
#   DeepSeek-V3-0324      — excellent multilingual + reasoning, high quota
#   o1-mini               — reasoning (slow, use only where needed)
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN", "")
LLM_MODEL_LIGHT = os.getenv("LLM_MODEL_LIGHT", "gpt-4o-mini")         # Step 2.5 ASR correct
LLM_MODEL       = os.getenv("LLM_MODEL",       "Phi-4")               # Step 4 normalization
LLM_MODEL_REASON= os.getenv("LLM_MODEL_REASON", "Phi-4")              # Steps 5-6 concepts/prereqs
LLM_BASE_URL    = "https://models.inference.ai.azure.com"

# Whisper config
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")  # good code-mixed accuracy
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")     # auto, cpu, cuda

# Audio extraction config
AUDIO_SAMPLE_RATE = 16000  # Whisper expects 16kHz
AUDIO_FORMAT = "wav"
