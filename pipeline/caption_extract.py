"""
YouTube Caption Extraction

Downloads auto-generated or manual captions from YouTube and converts
them into the same transcript format used by the Whisper STT module.

This is used as a fallback/alternative for languages where Whisper
performs poorly (e.g., Sanskrit, less-represented Indian languages).
"""
import json
import re
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi

from pipeline.config import get_video_dir


def download_captions(
    url: str,
    lang: str = "hi",
    video_id: str | None = None,
) -> dict:
    """
    Download YouTube auto-generated captions and convert to transcript format.

    Args:
        url: YouTube video URL.
        lang: Language code for captions (e.g., "hi" for Hindi, "sa" for Sanskrit,
              "ta" for Tamil, "te" for Telugu).
        video_id: Video ID for output directory. Auto-derived if None.

    Returns:
        Transcript data dict in the same format as Whisper output.
    """
    # Derive video_id
    if video_id is None:
        match = re.search(r'(?:v=|/v/|youtu\.be/)([\w-]{11})', url)
        video_id = match.group(1) if match else "unknown"

    video_dir = get_video_dir(video_id)

    print(f"[Captions] Downloading {lang} auto-captions for {video_id}...")

    ytt = YouTubeTranscriptApi()

    # Try requested language, then fall back to others
    try:
        result = ytt.fetch(video_id=video_id, languages=[lang])
    except Exception:
        print(f"[Captions] {lang} captions not available, trying auto-detect...")
        transcript_list = ytt.list(video_id=video_id)
        # Pick the first generated transcript available
        available = None
        for t in transcript_list:
            if t.is_generated:
                available = t
                break
        if available is None:
            # Pick any available transcript
            for t in transcript_list:
                available = t
                break
        if available is None:
            raise RuntimeError(f"No captions available for video {video_id}")
        print(f"[Captions] Using {available.language} ({available.language_code})")
        result = ytt.fetch(video_id=video_id, languages=[available.language_code])
        lang = available.language_code

    # Convert snippets to segments
    segments = []
    for i, snippet in enumerate(result.snippets):
        segments.append({
            "id": i,
            "start": round(snippet.start, 3),
            "end": round(snippet.start + snippet.duration, 3),
            "text": snippet.text.strip(),
            "words": [],
            "avg_logprob": 0.0,
            "no_speech_prob": 0.0,
        })

    if not segments:
        raise ValueError("No caption segments found.")

    full_text = " ".join(seg["text"] for seg in segments)
    duration = segments[-1]["end"] if segments else 0.0

    transcript_data = {
        "audio_file": str(video_dir / f"{video_id}.wav"),
        "model": f"youtube-auto-captions-{lang}",
        "language": lang,
        "language_probability": 1.0,
        "duration_seconds": round(duration, 2),
        "transcription_time_seconds": 0.0,
        "segments": segments,
        "full_text": full_text,
    }

    # Save transcript
    output_path = video_dir / f"{video_id}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)

    print(f"[Captions] Transcript saved to: {output_path}")
    print(f"[Captions] Segments: {len(segments)}")
    print(f"[Captions] Language: {lang}")
    print(f"[Captions] Duration: {duration:.1f}s")
    print(f"[Captions] Text length: {len(full_text)} chars")

    return transcript_data


def list_available_captions(url: str) -> dict[str, str]:
    """
    List available caption languages for a YouTube video.

    Returns:
        Dict of {language_code: language_name}.
    """
    match = re.search(r'(?:v=|/v/|youtu\.be/)([\w-]{11})', url)
    vid = match.group(1) if match else url

    ytt = YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id=vid)

    captions = {}
    for t in transcript_list:
        prefix = "[auto] " if t.is_generated else ""
        captions[t.language_code] = f"{prefix}{t.language}"

    return captions


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.caption_extract <youtube_url> [lang_code]")
        print("       python -m pipeline.caption_extract --list <youtube_url>")
        sys.exit(1)

    if sys.argv[1] == "--list":
        url = sys.argv[2]
        caps = list_available_captions(url)
        print(f"Available captions ({len(caps)}):")
        for code, name in sorted(caps.items()):
            print(f"  {code:5s} {name}")
    else:
        url = sys.argv[1]
        lang = sys.argv[2] if len(sys.argv) > 2 else "hi"
        result = download_captions(url, lang=lang)
        print(f"\nPreview: {result['full_text'][:500]}")
