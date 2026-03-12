"""
Step 2: Speech-to-Text using Sarvam AI API

Uses Sarvam AI's speech-to-text endpoint which:
  - Natively handles ALL major Indic languages + code-mixed content
  - Runs entirely server-side — no local model weights needed
  - Auto-detects language (best for code-mixed content)
  - Returns word-level timestamps

Previous transcribers preserved in:
  - transcribe_indic_conformer.py  (AI4Bharat local ONNX)
  - transcribe_faster_whisper.py   (faster-whisper)
"""
import json
import math
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from pipeline.config import get_video_dir

_SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
_CHUNK_DURATION_S = 29      # Sarvam STT limit is 30s per request
_TARGET_SR = 16000
_SEGMENT_GAP_S = 1.5


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


def _load_and_resample(path: str, target_sr: int = _TARGET_SR):
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        gcd = math.gcd(target_sr, sr)
        audio = resample_poly(audio, target_sr // gcd, sr // gcd).astype("float32")
    return audio, target_sr


def _audio_to_wav_bytes(audio, sr: int) -> bytes:
    import io
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _call_sarvam_stt(wav_bytes: bytes, api_key: str, language_code: str = "unknown") -> dict:
    import requests
    headers = {"api-subscription-key": api_key}
    data = {
        "model": "saarika:v2.5",
        "language_code": language_code,
        "with_timestamps": "true",
        "with_disfluencies": "false",
    }
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    resp = requests.post(_SARVAM_STT_URL, headers=headers, data=data, files=files, timeout=120)
    if not resp.ok:
        raise RuntimeError(f"Sarvam API error {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def _words_to_segments(words: list[dict]) -> list[dict]:
    if not words:
        return []
    segments: list[dict] = []
    current: list[dict] = []
    for w in words:
        if current and (w["start"] - current[-1]["end"]) >= _SEGMENT_GAP_S:
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


def transcribe(
    audio_path: str | Path,
    model_size: str = "saarika:v2.5",
    device: str = "auto",
    language: str | None = None,
    chunk_duration_s: int = _CHUNK_DURATION_S,
    # Legacy params kept for API compatibility
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
    batch_size: int = 8,
    chunk_length_s: int = 30,
) -> TranscriptResult:
    """
    Transcribe audio using the Sarvam AI speech-to-text API (saarika:v2).

    Handles all major Indic languages and code-mixed content automatically.
    Leave language=None for auto-detect (recommended for code-mixed).

    Args:
        audio_path: Path to audio file (WAV/MP3/FLAC/M4A etc.).
        language: BCP-47 code hint e.g. 'te-IN', 'hi-IN'. None = auto-detect.
        chunk_duration_s: Max seconds per API request (default 240 = 4 min).
    """
    import os
    from dotenv import load_dotenv

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=True)
    api_key = os.environ.get("SARVAM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "SARVAM_API_KEY not set. Add it to .env.\n"
            "Get a free key at https://dashboard.sarvam.ai"
        )

    lang_code = language if language else "unknown"
    print(f"[Step 2] Sarvam STT  |  language hint: {lang_code}")

    print(f"[Step 2] Loading audio: {audio_path.name}")
    audio, sr = _load_and_resample(str(audio_path), _TARGET_SR)
    duration = len(audio) / sr
    print(f"[Step 2] Duration: {duration:.1f}s")

    chunk_samples = chunk_duration_s * sr
    n_chunks = max(1, math.ceil(len(audio) / chunk_samples))
    print(f"[Step 2] Sending {n_chunks} chunk(s) to Sarvam API...")

    raw_segs: list[dict] = []   # {text, start, end}
    detected_languages: list[str] = []
    start_time = time.time()

    for i, chunk_start_sample in enumerate(range(0, len(audio), chunk_samples)):
        chunk = audio[chunk_start_sample: chunk_start_sample + chunk_samples]
        chunk_offset = chunk_start_sample / sr
        chunk_end = chunk_offset + len(chunk) / sr

        print(f"[Step 2] Chunk {i+1}/{n_chunks}  ({chunk_offset:.0f}s – {chunk_end:.0f}s)...")

        wav_bytes = _audio_to_wav_bytes(chunk, sr)
        response = _call_sarvam_stt(wav_bytes, api_key=api_key, language_code=lang_code)

        detected_lang = response.get("language_code", "unknown")
        if detected_lang and detected_lang not in detected_languages:
            detected_languages.append(detected_lang)

        # Sarvam returns sentence-level timestamps under response["timestamps"]
        ts = response.get("timestamps") or {}
        ts_words = ts.get("words", [])
        ts_starts = ts.get("start_time_seconds", [])
        ts_ends = ts.get("end_time_seconds", [])

        if ts_words:
            for idx, seg_text in enumerate(ts_words):
                seg_text = seg_text.strip()
                if not seg_text:
                    continue
                t0 = ts_starts[idx] if idx < len(ts_starts) else 0.0
                t1 = ts_ends[idx] if idx < len(ts_ends) else len(chunk) / sr
                raw_segs.append({
                    "text": seg_text,
                    "start": round(chunk_offset + t0, 3),
                    "end": round(chunk_offset + t1, 3),
                })
        else:
            text = response.get("transcript", "").strip()
            if text:
                raw_segs.append({
                    "text": text,
                    "start": round(chunk_offset, 3),
                    "end": round(chunk_end, 3),
                })

        print(f"[Step 2] Chunk {i+1} done — lang: {detected_lang}, segments so far: {len(raw_segs)}")

    elapsed = time.time() - start_time

    segments: list[Segment] = []
    for seg_dict in raw_segs:
        text = seg_dict["text"].strip()
        if not text:
            continue
        segments.append(Segment(
            id=len(segments),
            start=seg_dict["start"],
            end=seg_dict["end"],
            text=text,
            language=detected_languages[0] if detected_languages else None,
        ))

    primary_lang = detected_languages[0] if detected_languages else "unknown"
    if len(detected_languages) > 1:
        print(f"[Step 2] Languages detected: {detected_languages}")

    total_words = sum(len(s.text.split()) for s in segments)
    print(f"[Step 2] Done in {elapsed:.1f}s — {len(segments)} segments, "
          f"~{total_words} words, language: {primary_lang}")

    result = TranscriptResult(
        audio_file=str(audio_path),
        model="sarvam/saarika:v2.5",
        language=primary_lang,
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
        print("Usage: python -m pipeline.transcribe <audio_file> [language_code]")
        print("  language_code: BCP-47 e.g. te-IN, hi-IN  (omit for auto-detect)")
        sys.exit(1)
    lang_arg = sys.argv[2] if len(sys.argv) > 2 else None
    result = transcribe(sys.argv[1], language=lang_arg)
    print(f"\n=== Transcript ({len(result.segments)} segments) ===")
    for s in result.segments:
        print(f"[{s.start:.1f}s – {s.end:.1f}s] {s.text}")
    print(f"\nLanguage: {result.language}")
    print(f"Words: ~{sum(len(s.text.split()) for s in result.segments)}")
