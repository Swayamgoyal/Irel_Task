"""
Whisper Large Pipeline Orchestrator

Same pipeline as pipeline.main but uses OpenAI Whisper large-v3 (via
faster-whisper) for Step 2 instead of the Sarvam AI cloud API.

Outputs are written to  data_whisper/<video_id>/  so they can be
compared side-by-side with the Sarvam outputs in  data/<video_id>/.

Pipeline steps:
  Video/URL
    ↓
  Step 1: Audio Extraction       (ffmpeg + yt-dlp)
    ↓
  Step 2: Speech-to-Text         (Whisper large-v3 via faster-whisper)
    ↓
  Step 3: Language Detection     (Unicode script analysis)
    ↓
  Step 4: Normalization          (LLM_MODEL via GitHub Models)
    ↓
  Step 5: Concept Extraction     (LLM_MODEL_REASON via GitHub Models)
    ↓
  Step 6: Prerequisite Mapping   (LLM_MODEL_REASON via GitHub Models)
    ↓
  Steps 7-8: Graph + Output      (NetworkX + vis-network HTML)

Comparison usage:
  python -m pipeline.main        "https://youtu.be/VIDEO_ID"  # Sarvam → data/VIDEO_ID/
  python -m pipeline_whisper.main "https://youtu.be/VIDEO_ID"  # Whisper → data_whisper/VIDEO_ID/

Audio reuse:
  --skip-audio  looks for the WAV in data_whisper/<id>/ first, then
                falls back to data/<id>/ (Sarvam pipeline's download),
                so you don't need to re-download the same video.
"""
import sys
import json
import time
from pathlib import Path

# ── Patch pipeline.config BEFORE any pipeline.* module is imported ───────────
# All pipeline modules do  `from pipeline.config import get_video_dir`
# at their module top-level.  By patching the source object *before* those
# modules are first imported (they are all lazy-imported inside run_pipeline)
# every downstream save goes to  data_whisper/  instead of  data/.
import pipeline.config as _pipeline_config
from pipeline_whisper.config import (
    DATA_DIR        as _WHISPER_DATA_DIR,
    get_video_dir   as _whisper_get_video_dir,
    WHISPER_MODEL   as _WHISPER_MODEL_SIZE,
    LLM_MODEL_LIGHT as _LLM_MODEL_LIGHT,
    LLM_MODEL       as _LLM_MODEL,
    LLM_MODEL_REASON as _LLM_MODEL_REASON,
)

_pipeline_config.DATA_DIR    = _WHISPER_DATA_DIR
_pipeline_config.get_video_dir = _whisper_get_video_dir
# Also update WHISPER_MODEL so transcribe_faster_whisper picks up large-v3
# (we also pass it explicitly as model_size= below, but belt-and-suspenders)
_pipeline_config.WHISPER_MODEL = _WHISPER_MODEL_SIZE
# ─────────────────────────────────────────────────────────────────────────────

from pipeline_whisper.config import BASE_DIR, get_video_dir


def _derive_video_id(source: str) -> str:
    """Derive a video ID from a YouTube URL or local file path."""
    import re
    match = re.search(r'(?:v=|/v/|youtu\.be/)([\w-]{11})', source)
    if match:
        return match.group(1)
    return Path(source).stem


def _step_banner(n, title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'─' * 60}")


def run_pipeline(
    source: str,
    language: str | None = None,
    skip_audio: bool = False,
    skip_transcribe: bool = False,
    skip_correct: bool = False,
    skip_normalize: bool = False,
    skip_concepts: bool = False,
    skip_prereqs: bool = False,
    transcript_path: str | None = None,
) -> dict:
    """
    Run the full pipeline on a video source using Whisper large-v3.

    Args:
        source:           YouTube URL, local video, or local audio file.
        language:         ISO-639-1 language hint for Whisper (e.g. 'te', 'hi').
                          Leave None for auto-detect.
        skip_audio:       Use existing WAV — checks data_whisper/ first,
                          then falls back to data/ (Sarvam pipeline audio).
        skip_transcribe:  Use existing *_transcript.json in data_whisper/.
        skip_normalize:   Use existing *_normalized.json.
        skip_concepts:    Use existing *_concepts.json.
        skip_prereqs:     Use existing *_prerequisites.json.
        transcript_path:  Explicit path to transcript JSON (implies skip_transcribe).

    Returns:
        Final output dict.
    """
    start_time = time.time()
    print("=" * 60)
    print(f"  Code-Mixed Pedagogical Flow Extractor  [Whisper {_WHISPER_MODEL_SIZE}]")
    print("=" * 60)
    print(f"  Source : {source}")
    if language:
        print(f"  Lang   : {language}")
    print()

    video_id  = _derive_video_id(source)
    video_dir = get_video_dir(video_id)          # data_whisper/<video_id>/
    sarvam_dir = BASE_DIR / "data" / video_id    # data/<video_id>/ for audio reuse

    print(f"[Pipeline] Video ID  : {video_id}")
    print(f"[Pipeline] Output dir: {video_dir}")

    # ── Step 1 + 2: Audio + Transcription ────────────────────────────────────
    if transcript_path:
        print(f"\n[Pipeline] Loading transcript from: {transcript_path}")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

    elif skip_transcribe:
        t_path = video_dir / f"{video_id}_transcript.json"
        print(f"\n[Pipeline] Skipping Steps 1-2 — loading: {t_path.name}")
        with open(t_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)

    else:
        # Step 1: Audio — prefer reusing an existing WAV to avoid re-downloading
        whisper_wav = video_dir / f"{video_id}.wav"
        sarvam_wav  = sarvam_dir / f"{video_id}.wav"

        if skip_audio and whisper_wav.exists():
            _step_banner(1, "Audio Extraction — SKIPPED (whisper copy exists)")
            print(f"[Step 1] Using: {whisper_wav}")
            audio_path = whisper_wav

        elif skip_audio and sarvam_wav.exists():
            _step_banner(1, "Audio Extraction — SKIPPED (reusing Sarvam audio)")
            print(f"[Step 1] Reusing Sarvam pipeline audio: {sarvam_wav}")
            audio_path = sarvam_wav

        else:
            _step_banner(1, "Audio Extraction")
            from pipeline.audio_extract import process_input
            audio_path = process_input(source, loudnorm=True)

        # Step 2: STT — Whisper large-v3
        _step_banner(2, f"Speech-to-Text  (Whisper {_WHISPER_MODEL_SIZE})")
        from pipeline.transcribe_faster_whisper import transcribe
        transcript_result = transcribe(
            str(audio_path),
            model_size=_WHISPER_MODEL_SIZE,
            language=language,
        )
        transcript_data = transcript_result.to_dict()

    # ── Step 2.5: ASR Spelling Correction ────────────────────────────────────
    corrected_path = video_dir / f"{video_id}_transcript.json"
    if skip_correct:
        _step_banner("2.5", "ASR Correction — SKIPPED")
    else:
        _step_banner("2.5", f"ASR Spelling Correction  ({_LLM_MODEL_LIGHT})")
        from pipeline.asr_correct import correct_asr_transcript, save_corrected_transcript
        transcript_data = correct_asr_transcript(transcript_data)
        save_corrected_transcript(transcript_data, video_id)
        print(f"[Step 2.5] Saved corrected transcript → {corrected_path.name}")

    # ── Step 3: Language Detection ────────────────────────────────────────────
    _step_banner(3, "Language Detection & Code-Mix Analysis")
    from pipeline.language_detect import analyze_transcript
    language_profile    = analyze_transcript(transcript_data)
    lang_profile_dict   = language_profile.to_dict()

    # ── Step 4: Normalization ─────────────────────────────────────────────────
    norm_path = video_dir / f"{video_id}_normalized.json"
    if skip_normalize and norm_path.exists():
        _step_banner(4, "Normalization — SKIPPED (file exists)")
        print(f"[Step 4] Loading: {norm_path.name}")
        with open(norm_path, "r", encoding="utf-8") as f:
            normalized_data = json.load(f)
    else:
        _step_banner(4, f"Code-Mix Normalization  ({_LLM_MODEL})")
        from pipeline.normalize import normalize_and_standardize
        normalized_data = normalize_and_standardize(transcript_data, lang_profile_dict)

    # ── Step 5: Concept Extraction ────────────────────────────────────────────
    concepts_path = video_dir / f"{video_id}_concepts.json"
    if skip_concepts and concepts_path.exists():
        _step_banner(5, "Concept Extraction — SKIPPED (file exists)")
        print(f"[Step 5] Loading: {concepts_path.name}")
        with open(concepts_path, "r", encoding="utf-8") as f:
            concepts_data = json.load(f)
    else:
        _step_banner(5, f"Concept Extraction  ({_LLM_MODEL_REASON})")
        from pipeline.concept_extract import extract_concepts
        concepts_data = extract_concepts(normalized_data, transcript_data)

    # ── Step 6: Prerequisite Detection ───────────────────────────────────────
    prereqs_path = video_dir / f"{video_id}_prerequisites.json"
    if skip_prereqs and prereqs_path.exists():
        _step_banner(6, "Prerequisite Detection — SKIPPED (file exists)")
        print(f"[Step 6] Loading: {prereqs_path.name}")
        with open(prereqs_path, "r", encoding="utf-8") as f:
            prerequisites_data = json.load(f)
    else:
        _step_banner(6, f"Prerequisite Dependency Mapping  ({_LLM_MODEL_REASON})")
        from pipeline.prerequisite import detect_prerequisites
        prerequisites_data = detect_prerequisites(concepts_data, normalized_data)

    # ── Steps 7-8: Graph + Final Output ──────────────────────────────────────
    _step_banner("7-8", "Graph Construction & Output")
    from pipeline.graph_builder import (
        build_graph,
        generate_interactive_html,
        generate_static_png,
        generate_final_output,
    )
    graph         = build_graph(concepts_data, prerequisites_data)
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
    print(f"  ✓ Whisper pipeline complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"\nOutputs saved in: {video_dir}")
    print(f"  {video_id}_transcript.json    — Whisper {_WHISPER_MODEL_SIZE} transcript")
    print(f"  {video_id}_normalized.json    — English-normalized text + term map")
    print(f"  {video_id}_concepts.json      — extracted concepts + teaching flow")
    print(f"  {video_id}_prerequisites.json — dependency graph edges")
    print(f"  {video_id}_final_output.json  — complete machine-readable output")
    print(f"  prerequisite_graph.html       — interactive visual graph")
    print(f"  prerequisite_graph.png        — static graph image")
    print(f"\nCompare with Sarvam outputs in: data/{video_id}/")

    return final_output


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=f"Code-Mixed Pedagogical Flow Extractor — Whisper {_WHISPER_MODEL_SIZE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  # Full run — Whisper {_WHISPER_MODEL_SIZE} transcription → data_whisper/VIDEO_ID/
  python -m pipeline_whisper.main https://www.youtube.com/watch?v=VIDEO_ID

  # Force Telugu language for Whisper
  python -m pipeline_whisper.main https://youtu.be/VIDEO_ID --language te

  # Reuse audio already downloaded by the Sarvam pipeline (no re-download)
  python -m pipeline_whisper.main https://youtu.be/VIDEO_ID --skip-audio

  # Resume after transcription is done
  python -m pipeline_whisper.main https://youtu.be/VIDEO_ID --skip-audio --skip-transcribe

  # Full resume — only re-run graphs
  python -m pipeline_whisper.main https://youtu.be/VIDEO_ID \\
      --skip-audio --skip-transcribe --skip-normalize --skip-concepts --skip-prereqs

  # Compare both pipelines on the same video:
  python -m pipeline.main         https://youtu.be/VIDEO_ID   # Sarvam  → data/VIDEO_ID/
  python -m pipeline_whisper.main https://youtu.be/VIDEO_ID   # Whisper → data_whisper/VIDEO_ID/
"""
    )
    parser.add_argument(
        "source",
        help="YouTube URL, local video file, or local audio file",
    )
    parser.add_argument(
        "--language", "-l", type=str, default=None, metavar="LANG",
        help=(
            "ISO-639-1 language hint for Whisper (e.g. te, hi, ta, ml). "
            "Default: auto-detect from first 30 s of audio."
        ),
    )
    parser.add_argument(
        "--skip-audio", action="store_true",
        help=(
            "Skip Step 1: use existing WAV. "
            "Checks data_whisper/<id>/ first, then falls back to data/<id>/ "
            "(Sarvam pipeline audio) so no re-download is needed."
        ),
    )
    parser.add_argument(
        "--skip-transcribe", action="store_true",
        help="Skip Steps 1-2: use existing *_transcript.json in data_whisper/<video_id>/",
    )
    parser.add_argument(
        "--skip-correct", action="store_true",
        help="Skip Step 2.5: skip ASR spelling correction pass",
    )
    parser.add_argument(
        "--skip-normalize", action="store_true",
        help="Skip Step 4: use existing *_normalized.json",
    )
    parser.add_argument(
        "--skip-concepts", action="store_true",
        help="Skip Step 5: use existing *_concepts.json",
    )
    parser.add_argument(
        "--skip-prereqs", action="store_true",
        help="Skip Step 6: use existing *_prerequisites.json",
    )
    parser.add_argument(
        "--transcript", type=str, default=None, metavar="PATH",
        help="Path to an existing transcript JSON (implies --skip-transcribe)",
    )

    args = parser.parse_args()

    run_pipeline(
        source=args.source,
        language=args.language,
        skip_audio=args.skip_audio,
        skip_transcribe=args.skip_transcribe,
        skip_correct=args.skip_correct,
        skip_normalize=args.skip_normalize,
        skip_concepts=args.skip_concepts,
        skip_prereqs=args.skip_prereqs,
        transcript_path=args.transcript,
    )


if __name__ == "__main__":
    main()
