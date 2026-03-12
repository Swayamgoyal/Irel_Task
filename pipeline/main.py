"""
Main Pipeline Orchestrator

Runs the complete Code-Mixed Pedagogical Flow Extractor pipeline:

  Video/URL
    ↓
  Step 1: Audio Extraction (ffmpeg + yt-dlp)
    ↓
  Step 2: Speech-to-Text (Sarvam AI, code-mixed aware)
    ↓
  Step 3: Language Detection & Code-Mix Analysis (AI4Bharat-inspired)
    ↓
  Step 4: Code-Mix Normalization & Standardization (Gemini)
    ↓
  Step 5: Concept Extraction (Gemini)
    ↓
  Step 6: Prerequisite Dependency Detection (Gemini)
    ↓
  Steps 7-8: Graph Construction & Output (NetworkX + pyvis)
"""
import sys
import json
import time
from pathlib import Path

from pipeline.config import get_video_dir


def _derive_video_id(source: str) -> str:
    """Derive a video ID from web URL or local file path."""
    import re
    match = re.search(r'(?:v=|/v/|youtu\.be/)([\w-]{11})', source)
    if match:
        return match.group(1)
    return Path(source).stem


def _is_youtube_url(source: str) -> bool:
    """Check if source is a YouTube URL."""
    return "youtube.com" in source or "youtu.be" in source


def run_pipeline(
    source: str,
    skip_audio: bool = False,
    skip_transcribe: bool = False,
    transcript_path: str | None = None,
    whisper_lang: str | None = None,
) -> dict:
    """
    Run the full pipeline on a video source.

    Default flow: Video → Audio Extraction → Whisper STT → NLP pipeline.
    Whisper always transcribes in the original language (never translates).

    Args:
        source: Video file path or YouTube URL.
        skip_audio: Skip Step 1 if audio already extracted.
        skip_transcribe: Skip Step 2 if transcript already exists.
        transcript_path: Path to existing transcript JSON (skips Steps 1-2).
        whisper_lang: Force Whisper language (e.g., "ta", "te", "hi").
                      If None, Whisper auto-detects the language.

    Returns:
        Final output dict with all pipeline results.
    """
    start_time = time.time()
    print("=" * 60)
    print("  Code-Mixed Pedagogical Flow Extractor")
    print("=" * 60)
    print(f"Source: {source}\n")

    video_id = _derive_video_id(source)
    video_dir = get_video_dir(video_id)
    print(f"[Pipeline] Video ID: {video_id}")
    print(f"[Pipeline] Output dir: {video_dir}\n")

    # ── Steps 1-2: Audio Extraction + Speech-to-Text ──
    if transcript_path:
        print("[Pipeline] Using provided transcript")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

    elif skip_transcribe:
        print("[Pipeline] Using existing transcript")
        t_path = video_dir / f"{video_id}_transcript.json"
        with open(t_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

    else:
        # Step 1: Audio Extraction
        if not skip_audio:
            from pipeline.audio_extract import process_input
            print("\n── Step 1: Audio Extraction ──")
            audio_path = process_input(source)
        else:
            audio_path = Path(source)
            print(f"[Pipeline] Using existing audio: {audio_path}")

        # Step 2: Whisper Speech-to-Text
        from pipeline.transcribe import transcribe
        print("\n── Step 2: Speech-to-Text (Sarvam AI) ──")
        transcript_result = transcribe(str(audio_path), language=whisper_lang)
        transcript_data = transcript_result.to_dict()

    # ── Step 3: Language Detection ──
    from pipeline.language_detect import analyze_transcript
    print("\n── Step 3: Language Detection & Code-Mix Analysis ──")
    language_profile = analyze_transcript(transcript_data)
    lang_profile_dict = language_profile.to_dict()

    # ── Step 4: Normalization & Standardization ──
    from pipeline.normalize import normalize_and_standardize
    print("\n── Step 4: Code-Mix Normalization & Standardization ──")
    normalized_data = normalize_and_standardize(transcript_data, lang_profile_dict)

    # ── Step 5: Concept Extraction ──
    from pipeline.concept_extract import extract_concepts
    print("\n── Step 5: Concept Extraction ──")
    concepts_data = extract_concepts(normalized_data, transcript_data)

    # ── Step 6: Prerequisite Detection ──
    from pipeline.prerequisite import detect_prerequisites
    print("\n── Step 6: Prerequisite Dependency Detection ──")
    prerequisites_data = detect_prerequisites(concepts_data, normalized_data)

    # ── Steps 7-8: Graph Construction & Output ──
    from pipeline.graph_builder import (
        build_graph,
        generate_interactive_html,
        generate_static_png,
        generate_final_output,
    )
    print("\n── Steps 7-8: Graph Construction & Output ──")
    graph = build_graph(concepts_data, prerequisites_data)
    lecture_title = concepts_data.get("lecture_topic", "")
    generate_interactive_html(graph, video_id=video_id, title=lecture_title)
    generate_static_png(graph, video_id=video_id)

    final_output = generate_final_output(
        transcript_data=transcript_data,
        language_profile=lang_profile_dict,
        normalized_data=normalized_data,
        concepts_data=concepts_data,
        prerequisites_data=prerequisites_data,
        graph=graph,
        video_source=source,
        video_id=video_id,
    )

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\nOutputs in: {video_dir}")
    print(f"  - Final JSON:      *_final_output.json")
    print(f"  - Interactive Map:  prerequisite_graph.html")
    print(f"  - Static Graph:    prerequisite_graph.png")
    print(f"  - Normalized Text: *_normalized.json")
    print(f"  - Concepts:        *_concepts.json")
    print(f"  - Prerequisites:   *_prerequisites.json")

    return final_output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Code-Mixed Pedagogical Flow Extractor Pipeline"
    )
    parser.add_argument(
        "source",
        help="Video file path, YouTube URL, or audio file path",
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help="Skip audio extraction (source is already an audio file)",
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="Skip transcription (transcript already exists in data/transcripts/)",
    )
    parser.add_argument(
        "--transcript", type=str, default=None,
        help="Path to existing transcript JSON (skips Steps 1-2)",
    )
    parser.add_argument(
        "--whisper-lang", type=str, default=None,
        help="Force Whisper language (e.g., ta, te, hi) to prevent auto-translation",
    )

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        skip_audio=args.skip_audio,
        skip_transcribe=args.skip_transcribe,
        transcript_path=args.transcript,
        whisper_lang=args.whisper_lang,
    )


if __name__ == "__main__":
    main()
