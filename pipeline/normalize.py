"""
Step 4: Code-Mix Normalization & Linguistic Standardization

Takes the raw code-mixed transcript and:
  1. Normalizes transliterated Indic words (e.g., "koushin" → "question")
  2. Maps colloquial terms to standard academic terminology
  3. Translates remaining Indic-language portions to English
  4. Produces a clean, standardized English transcript

Uses Gemini for its superior Indian language understanding.
"""
import json
from pathlib import Path

from pipeline.llm_client import call_gemini_json, call_gemini
from pipeline.config import get_video_dir, LLM_MODEL as _MODEL

SYSTEM_INSTRUCTION = """You are an expert in Indian code-mixed languages (Hinglish, Tanglish, etc.)
and computer science / engineering / biology / physics terminology.

Your task is to normalize and standardize educational transcripts that may be:
- Code-mixed (Hindi-English, Tamil-English, Telugu-English, Bengali-English, etc.)
- Auto-translated from an Indian language to English by Whisper ASR
- Already in English with some transcription errors
- Containing residual ASR phonetic spelling errors in native script

Rules:
1. Convert ALL text to clean, standard academic English.
2. Map colloquial/transliterated Indic terms to proper English equivalents.
3. Fix Whisper ASR errors — both English typos AND native-script phonetic misspellings
   that survived the correction step (e.g. "বইখিস্টার" → "characteristics",
   "কোকবিফাজন" → "cell division", "বিফিন্ণ" → "different").
4. Fix awkward auto-translations — make them sound natural and academically correct.
5. Keep the meaning and teaching flow intact — do NOT summarize or skip content.
6. Break the output into logical paragraphs aligned with teaching flow.
7. Mark where the teacher transitions between sub-topics with [TOPIC_SHIFT] markers.
8. If the text is already clean English, still fix grammar issues and add topic shift markers.
9. Infer the correct academic term from context when a word is clearly a phonetic corruption
   (e.g. if the topic is biology and you see "growth and development", fill in standard terms).
"""


def normalize_and_standardize(
    transcript_data: dict,
    language_profile: dict | None = None,
) -> dict:
    """
    Normalize code-mixed transcript to standard English.

    Processes in chunks to handle long transcripts within context limits.

    Args:
        transcript_data: Parsed transcript JSON from Step 2.
        language_profile: Language analysis from Step 3 (optional, adds context).

    Returns:
        Dict with normalized text and metadata.
    """
    full_text = transcript_data.get("full_text", "")
    segments = transcript_data.get("segments", [])
    detected_lang = transcript_data.get("language", "unknown")

    # Build context about the languages
    lang_context = ""
    if language_profile:
        langs = language_profile.get("languages_detected", [])
        dist = language_profile.get("overall_distribution", {})
        lang_context = f"\nDetected languages: {langs}\nDistribution: {dist}\n"
        code_mix = language_profile.get("code_mix_ratio", 0)
        if code_mix == 0:
            lang_context += (
                "\nNote: The transcript appears to be already in English "
                "(possibly auto-translated by Whisper from the original language). "
                "Focus on fixing any awkward translations and transcription errors.\n"
            )

    # Filter out empty segments and build chunks (~3000 chars each)
    valid_segments = [seg for seg in segments if seg.get("text", "").strip()]
    
    if not valid_segments:
        # Fallback: use full_text directly
        valid_segments = [{"id": 0, "start": 0, "end": 0, "text": full_text}]
    
    chunk_size = 3000
    chunks = []
    current_chunk = []
    current_len = 0

    for seg in valid_segments:
        text = seg["text"].strip()
        if current_len + len(text) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_len = 0
        current_chunk.append(seg)
        current_len += len(text)

    if current_chunk:
        chunks.append(current_chunk)

    print(f"[Step 4] Processing {len(chunks)} chunks for normalization...")

    normalized_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_text = "\n".join(seg["text"].strip() for seg in chunk)
        time_range = f"{chunk[0]['start']:.1f}s - {chunk[-1]['end']:.1f}s"

        prompt = f"""Normalize and standardize this code-mixed educational transcript chunk.
{lang_context}
Time range: {time_range}
Chunk {i + 1}/{len(chunks)}:

---
{chunk_text}
---

Return JSON with this exact structure:
{{
  "normalized_text": "The full normalized English text for this chunk",
  "topic_shifts": ["brief description of each topic transition detected"],
  "technical_terms_mapped": {{"original_term": "standardized_term"}}
}}"""

        print(f"[Step 4] Normalizing chunk {i + 1}/{len(chunks)} ({time_range})...")

        result = call_gemini_json(prompt=prompt, system_instruction=SYSTEM_INSTRUCTION, model=_MODEL)
        result["chunk_index"] = i
        result["time_range"] = time_range
        result["original_segments"] = [seg["id"] for seg in chunk]
        normalized_chunks.append(result)

    # Combine all chunks
    full_normalized = "\n\n".join(
        chunk.get("normalized_text", "") for chunk in normalized_chunks
    )

    # Collect all term mappings
    all_mappings = {}
    all_topic_shifts = []
    for chunk in normalized_chunks:
        all_mappings.update(chunk.get("technical_terms_mapped", {}))
        all_topic_shifts.extend(chunk.get("topic_shifts", []))

    output = {
        "original_language": detected_lang,
        "normalized_text": full_normalized,
        "technical_terms_mapped": all_mappings,
        "topic_shifts": all_topic_shifts,
        "chunks": normalized_chunks,
        "total_chunks": len(normalized_chunks),
        "_audio_file": transcript_data.get("audio_file", "unknown"),
    }

    # Save output
    stem = Path(transcript_data.get("audio_file", "unknown")).stem
    video_dir = get_video_dir(stem)
    output_path = video_dir / f"{stem}_normalized.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"[Step 4] Normalized transcript saved to: {output_path}")
    print(f"[Step 4] Terms mapped: {len(all_mappings)}")
    print(f"[Step 4] Topic shifts found: {len(all_topic_shifts)}")

    return output


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.normalize <transcript.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    result = normalize_and_standardize(data)
    print(f"\n--- Normalized Preview (first 500 chars) ---")
    print(result["normalized_text"][:500])
