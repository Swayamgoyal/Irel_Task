"""
Step 2.5: ASR Spelling Correction

Whisper is good at hearing speech but frequently makes phonetic spelling
mistakes when writing Indic scripts, especially in code-mixed lectures:

  Heard correctly → spelled wrong:
    "বইখিস্টার"  → "বৈশিষ্ট্য"   (characteristics)
    "বিফিন্ণ"    → "বিভিন্ন"      (different)
    "কোকবিফাজন"  → "কোষ বিভাজন"  (cell division)

This step runs gpt-4o-mini over the raw transcript segments and asks it to
fix ONLY spelling/script errors — it does NOT translate, summarize, or
reorder content.  Timestamps are preserved exactly.

The corrected transcript is saved as  *_transcript.json  (overwrite) so all
downstream steps (language detect, normalisation, concepts) operate on
cleaner input.
"""
import json
from pathlib import Path

from pipeline.llm_client import call_gemini_json
from pipeline.config import get_video_dir, LLM_MODEL_LIGHT as _MODEL

# Map of ISO-639-1 code → language name for prompt clarity
_LANG_NAMES = {
    "bn": "Bengali/Assamese",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "sa": "Sanskrit",
    "or": "Odia",
    "ne": "Nepali",
    "si": "Sinhala",
    "as": "Assamese",
    "en": "English",
}

_SYSTEM = """You are an expert ASR (automatic speech recognition) post-processor
specialising in Indian languages written in their native scripts.

Whisper hears speech correctly but often produces phonetic spelling errors.
Your ONLY job is to fix those spelling errors.

STRICT RULES:
1. Fix phonetic spelling errors to correct native-script spellings.
2. Do NOT translate anything — output stays in the same language as input.
3. Do NOT add, remove, or reorder words.
4. Do NOT paraphrase or rephrase.
5. Keep all English words exactly as they are.
6. Keep numbers, punctuation, and symbols exactly as they are.
7. If a word looks correct, leave it unchanged.

Return ONLY the JSON structure requested — no explanations."""


# Batch size: number of segments per LLM call (balance context vs API cost)
_BATCH = 25


def correct_asr_transcript(transcript_data: dict) -> dict:
    """
    Fix ASR phonetic spelling errors in raw transcript segments.

    Processes segments in batches, asking gpt-4o-mini to correct only
    spelling mistakes in the native script.  Timestamps are untouched.

    Args:
        transcript_data: Parsed transcript dict (from transcribe_faster_whisper).

    Returns:
        Updated transcript dict with corrected segment texts and full_text.
    """
    import copy
    data = copy.deepcopy(transcript_data)

    detected_lang = data.get("language", "unknown")
    lang_name = _LANG_NAMES.get(detected_lang, detected_lang)
    segments = data.get("segments", [])

    # Nothing to do for pure English or empty transcripts
    if not segments or detected_lang == "en":
        print(f"[Step 2.5] Skipping correction — language: {detected_lang}")
        return data

    # Build batches of segment indices
    idxs = [i for i, s in enumerate(segments) if s.get("text", "").strip()]
    batches = [idxs[i:i + _BATCH] for i in range(0, len(idxs), _BATCH)]
    print(f"[Step 2.5] Correcting {len(idxs)} segments in {len(batches)} batches "
          f"(lang={detected_lang} / {lang_name})")

    total_changed = 0
    for b_num, batch_idxs in enumerate(batches, 1):
        # Build input list for this batch
        seg_list = [
            {"id": segments[i]["id"], "text": segments[i]["text"].strip()}
            for i in batch_idxs
        ]

        prompt = f"""Fix ASR phonetic spelling errors in these {lang_name} lecture transcript segments.

Language: {lang_name} (ISO code: {detected_lang}), mixed with English technical terms.
The lecturer is teaching a science/biology/technology topic.

Each segment has an "id" and "text". Correct ONLY the {lang_name} spelling errors.
Keep every English word exactly as-is. Keep numbers and punctuation unchanged.

Input segments:
{json.dumps(seg_list, ensure_ascii=False, indent=2)}

Return JSON with this exact structure:
{{
  "corrected": [
    {{"id": <same id as input>, "text": "<corrected text>"}},
    ...
  ]
}}

Include ALL input segments in the output, even if unchanged."""

        try:
            result = call_gemini_json(prompt=prompt, system_instruction=_SYSTEM, model=_MODEL)
            corrected_list = result.get("corrected", [])
        except Exception as e:
            print(f"[Step 2.5]   batch {b_num}/{len(batches)} failed: {e} — keeping originals")
            continue

        # Build a lookup by segment id
        corrected_map = {item["id"]: item["text"] for item in corrected_list}

        for i in batch_idxs:
            seg = segments[i]
            orig = seg["text"].strip()
            fixed = corrected_map.get(seg["id"], orig).strip()
            if fixed and fixed != orig:
                segments[i]["text"] = fixed
                total_changed += 1

        print(f"[Step 2.5]   batch {b_num}/{len(batches)} done  "
              f"(changed so far: {total_changed})")

    # Rebuild full_text from corrected segments
    data["segments"] = segments
    data["full_text"] = " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())

    print(f"[Step 2.5] Correction complete — {total_changed}/{len(idxs)} segments updated")
    return data


def save_corrected_transcript(transcript_data: dict, video_id: str) -> Path:
    """Save corrected transcript, overwriting the original _transcript.json."""
    video_dir = get_video_dir(video_id)
    output_path = video_dir / f"{video_id}_transcript.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(transcript_data, f, ensure_ascii=False, indent=2)
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.asr_correct <transcript.json>")
        sys.exit(1)
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    corrected = correct_asr_transcript(data)
    out = Path(sys.argv[1])
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)
    print(f"Saved corrected transcript to: {out}")
