"""
Step 2: Speech-to-Text using IndicConformer (ai4bharat/indic-conformer-600m-multilingual)

Uses AI4Bharat's IndicConformer ONNX model via HuggingFace Transformers for accurate
transcription of Indic languages and code-mixed content.

Key features:
  - Fine-tuned specifically on 22 Indic languages (hi, te, ta, ml, kn, bn, mr, gu, pa, ur, ...)
  - CTC + RNNT decoding with native word-level timestamps
  - Chunked inference for long audio (>30s)
  - Direct ONNX inference via onnxruntime (fast, no GPU memory overhead for decode)

The old faster-whisper implementation is preserved in transcribe_faster_whisper.py.

Output: Structured transcript with timestamps, detected languages, and
        word-level detail saved as JSON.
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from pipeline.config import get_video_dir

# HuggingFace model ID for AI4Bharat's IndicConformer
INDIC_WHISPER_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

# Languages supported by IndicConformer (ISO 639-1 codes)
SUPPORTED_LANGS = {
    "as", "bn", "brx", "doi", "gu", "hi", "kn", "kok", "ks", "mai",
    "ml", "mni", "mr", "ne", "or", "pa", "sa", "sat", "sd", "ta", "te", "ur",
}

# Target sample rate expected by the model
_TARGET_SR = 16000

# Silence gap (seconds) between words that triggers a new segment
_SEGMENT_GAP_S = 1.0


def _load_audio(path: str, target_sr: int = _TARGET_SR):
    """Load audio as float32 numpy array at target_sr. Returns (array, sr)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if sr != target_sr:
        from scipy.signal import resample_poly
        import math
        gcd = math.gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype("float32")

    return audio, target_sr


def _words_to_segments(words: list[dict]) -> list[dict]:
    """
    Group flat word dicts (with 'word', 'start', 'end') into sentence-like
    segments by splitting on silence gaps >= _SEGMENT_GAP_S.
    """
    if not words:
        return []

    segments: list[dict] = []
    current: list[dict] = []

    for w in words:
        if current:
            gap = w["start"] - current[-1]["end"]
            if gap >= _SEGMENT_GAP_S:
                segments.append(_flush(current))
                current = []
        current.append(w)

    if current:
        segments.append(_flush(current))

    return segments


def _flush(words: list[dict]) -> dict:
    return {
        "text": " ".join(w["word"] for w in words),
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    language: str | None = None
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptResult:
    audio_file: str
    model: str
    language: str
    language_probability: float
    duration_seconds: float
    transcription_time_seconds: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["full_text"] = self.full_text
        return d

    def save(self, output_path: Path | None = None) -> Path:
        if output_path is None:
            stem = Path(self.audio_file).stem
            video_dir = get_video_dir(stem)
            output_path = video_dir / f"{stem}_transcript.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return output_path


def transcribe(
    audio_path: str | Path,
    model_size: str = INDIC_WHISPER_MODEL_ID,
    device: str = "auto",
    language: str | None = None,
    chunk_length_s: int = 30,
    # Legacy params kept for API compatibility — not used by this backend
    batch_size: int = 8,
    beam_size: int = 5,
    best_of: int = 5,
    patience: float = 1.0,
    temperature: tuple = (0.0,),
    vad_filter: bool = True,
    word_timestamps: bool = True,
    condition_on_previous_text: bool = False,
    compression_ratio_threshold: float = 2.4,
    log_prob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    initial_prompt: str | None = None,
) -> TranscriptResult:
    """
    Transcribe audio using AI4Bharat's IndicConformer ONNX model.

    The model is downloaded from HuggingFace on first use (~2-3 GB) and cached.
    It uses CTC decoding with native word-level timestamps.

    Args:
        audio_path: Path to audio file (WAV/FLAC/MP3 at any sample rate).
        model_size: HuggingFace model ID, or legacy size string ('medium' etc.)
                    — if no '/' present, uses the default INDIC_WHISPER_MODEL_ID.
        language: ISO 639-1 code of the spoken language (e.g. 'te', 'hi', 'ta').
                  Defaults to 'hi' if not specified.
        chunk_length_s: Seconds per chunk for long-audio processing.

    Returns:
        TranscriptResult with segments, word timestamps, and metadata.
    """
    import os
    import torch
    import numpy as np
    from transformers import AutoModel
    from dotenv import load_dotenv

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model_id = model_size if "/" in model_size else INDIC_WHISPER_MODEL_ID

    # ── Auth ──────────────────────────────────────────────────────────────────
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
    hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN") or None
    if hf_token:
        print(f"[Step 2] Using HF token from .env")
        from huggingface_hub import login as hf_login
        hf_login(token=hf_token, add_to_git_credential=False)
    else:
        print(f"[Step 2] No HF token in .env — trying cached credentials")

    # ── Language ──────────────────────────────────────────────────────────────
    lang = (language or "hi").lower()
    if lang not in SUPPORTED_LANGS:
        print(f"[Step 2] Warning: '{lang}' not in supported langs, falling back to 'hi'")
        lang = "hi"
    print(f"[Step 2] Language: {lang}")

    # ── Device ────────────────────────────────────────────────────────────────
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Step 2] Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    # The custom from_pretrained calls snapshot_download internally; cached files
    # are reused automatically. FRAME_DURATION_MS=0.08 suppresses the accuracy warning.
    import os as _os
    _os.environ.setdefault("HF_HUB_HTTP_TIMEOUT", "120")

    print(f"[Step 2] Loading model: {model_id}  (uses cache after first download)")
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        token=hf_token,
        FRAME_DURATION_MS=0.08,   # 80ms per conformer encoder frame
    )
    model.eval()
    print(f"[Step 2] Model loaded")

    # ── Load & resample audio ─────────────────────────────────────────────────
    print(f"[Step 2] Loading audio: {audio_path.name}")
    audio, sr = _load_audio(str(audio_path), _TARGET_SR)
    duration = len(audio) / sr
    print(f"[Step 2] Duration: {duration:.1f}s, sample rate: {sr} Hz")

    # ── Chunked inference ─────────────────────────────────────────────────────
    chunk_samples = chunk_length_s * sr
    all_words: list[dict] = []
    start_time = time.time()

    n_chunks = max(1, int(np.ceil(len(audio) / chunk_samples)))
    print(f"[Step 2] Transcribing in {n_chunks} chunk(s) of {chunk_length_s}s...")

    with torch.no_grad():
        for i, chunk_start in enumerate(range(0, len(audio), chunk_samples)):
            chunk = audio[chunk_start: chunk_start + chunk_samples]
            wav_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0)
            chunk_offset = chunk_start / sr

            out = model(wav_tensor, lang=lang, decoding="ctc", compute_timestamps="w")

            if isinstance(out, tuple):
                _, word_ts = out
                # word_ts is list[list[(word, t0, t1)]] — one list per batch element
                for word, t0, t1 in word_ts[0]:
                    word = word.strip()
                    if not word or t0 is None or t1 is None:
                        continue
                    all_words.append({
                        "word": word,
                        "start": round(chunk_offset + t0, 3),
                        "end": round(chunk_offset + t1, 3),
                        "probability": 1.0,
                    })
            else:
                # Fallback: no timestamps available — create single word entry
                text = str(out).strip()
                if text:
                    all_words.append({
                        "word": text,
                        "start": round(chunk_offset, 3),
                        "end": round(chunk_offset + len(chunk) / sr, 3),
                        "probability": 1.0,
                    })

            print(f"[Step 2] Chunk {i+1}/{n_chunks} done, words so far: {len(all_words)}")

    elapsed = time.time() - start_time

    # ── Build segments ─────────────────────────────────────────────────────────
    seg_dicts = _words_to_segments(all_words)

    segments: list[Segment] = []
    for seg_dict in seg_dicts:
        text = seg_dict["text"].strip()
        if not text:
            continue
        words = [
            Word(
                word=w["word"],
                start=w["start"],
                end=w["end"],
                probability=w.get("probability", 1.0),
            )
            for w in seg_dict.get("words", [])
        ]
        segments.append(Segment(
            id=len(segments),
            start=seg_dict["start"],
            end=seg_dict["end"],
            text=text,
            words=words,
        ))

    print(f"[Step 2] Done in {elapsed:.1f}s — {len(segments)} segments, {len(all_words)} words")

    result = TranscriptResult(
        audio_file=str(audio_path),
        model=model_id,
        language=lang,
        language_probability=1.0,
        duration_seconds=round(duration, 2),
        transcription_time_seconds=round(elapsed, 2),
        segments=segments,
    )

    output_path = result.save()
    print(f"[Step 2] Transcript saved to: {output_path}")
    return result


def load_transcript(path: str | Path) -> TranscriptResult:
    """Load a previously saved transcript from JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg_data in data.get("segments", []):
        words = [Word(**w) for w in seg_data.get("words", [])]
        seg = Segment(
            id=seg_data["id"],
            start=seg_data["start"],
            end=seg_data["end"],
            text=seg_data["text"],
            language=seg_data.get("language"),
            avg_logprob=seg_data.get("avg_logprob", 0.0),
            no_speech_prob=seg_data.get("no_speech_prob", 0.0),
            words=words,
        )
        segments.append(seg)

    return TranscriptResult(
        audio_file=data["audio_file"],
        model=data["model"],
        language=data["language"],
        language_probability=data["language_probability"],
        duration_seconds=data["duration_seconds"],
        transcription_time_seconds=data["transcription_time_seconds"],
        segments=segments,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.transcribe <audio_file.wav> [language_code]")
        sys.exit(1)
    lang_arg = sys.argv[2] if len(sys.argv) > 2 else None
    result = transcribe(sys.argv[1], language=lang_arg)
    print(f"\n--- Full Transcript (first 500 chars) ---")
    print(result.full_text[:500])

import re
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from pipeline.config import get_video_dir

# HuggingFace model ID for AI4Bharat's IndicConformer
INDIC_WHISPER_MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

# Map ISO 639-1 codes → Whisper language names (used in generate_kwargs)
_ISO_TO_WHISPER_LANG = {
    "hi": "hindi",
    "te": "telugu",
    "ta": "tamil",
    "ml": "malayalam",
    "kn": "kannada",
    "bn": "bengali",
    "mr": "marathi",
    "gu": "gujarati",
    "pa": "punjabi",
    "ur": "urdu",
    "or": "odia",
    "sa": "sanskrit",
    "ne": "nepali",
    "si": "sinhala",
    "en": "english",
}

# Silence gap (seconds) between words that triggers a new segment
_SEGMENT_GAP_S = 1.0


def _words_to_segments(word_chunks: list[dict]) -> list[dict]:
    """
    Group flat word-level chunks from the transformers pipeline into
    sentence-like segments, splitting on gaps >= _SEGMENT_GAP_S.

    Each chunk has: {"text": str, "timestamp": (start, end)}
    Returns a list of dicts: {"text", "start", "end", "words"}
    """
    if not word_chunks:
        return []

    segments: list[dict] = []
    current_words: list[dict] = []

    for chunk in word_chunks:
        ts = chunk.get("timestamp") or (None, None)
        start, end = ts[0], ts[1]
        word_text = chunk.get("text", "").strip()
        if not word_text:
            continue

        if current_words:
            prev_end = current_words[-1]["end"]
            # Gap detected → flush current group as a new segment
            if start is not None and prev_end is not None and (start - prev_end) >= _SEGMENT_GAP_S:
                segments.append(_flush_segment(current_words))
                current_words = []

        current_words.append({
            "word": word_text,
            "start": start if start is not None else 0.0,
            "end": end if end is not None else 0.0,
            "probability": 1.0,  # transformers pipeline doesn't expose per-word probs
        })

    if current_words:
        segments.append(_flush_segment(current_words))

    return segments


def _flush_segment(words: list[dict]) -> dict:
    text = " ".join(w["word"] for w in words)
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": words,
    }


@dataclass
class Word:
    """A single word with timing info."""
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    """A transcript segment (roughly one sentence/phrase)."""
    id: int
    start: float
    end: float
    text: str
    language: str | None = None
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    words: list[Word] = field(default_factory=list)


@dataclass
class TranscriptResult:
    """Complete transcription output."""
    audio_file: str
    model: str
    language: str
    language_probability: float
    duration_seconds: float
    transcription_time_seconds: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return " ".join(seg.text.strip() for seg in self.segments)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["full_text"] = self.full_text
        return d

    def save(self, output_path: Path | None = None) -> Path:
        if output_path is None:
            stem = Path(self.audio_file).stem
            video_dir = get_video_dir(stem)
            output_path = video_dir / f"{stem}_transcript.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return output_path


