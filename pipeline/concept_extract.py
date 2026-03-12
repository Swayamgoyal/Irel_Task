"""
Step 5: Concept Extraction

Uses Gemini to extract core technical concepts from the normalized transcript.
Identifies:
  - Concept name (standard academic term)
  - Description (how the teacher explained it)
  - Timestamp range (when it was discussed)
  - Importance level (core vs supporting concept)
  - Related colloquial terms used by the teacher
"""
import json
from pathlib import Path

from pipeline.llm_client import call_gemini_json
from pipeline.config import get_video_dir, LLM_MODEL_REASON as _MODEL

SYSTEM_INSTRUCTION = """You are an expert in extracting structured knowledge from educational content.
You specialize in computer science, physics, and engineering pedagogy.

Your task is to identify and extract ALL technical concepts taught in a lecture transcript.

Rules:
1. Extract EVERY distinct technical concept, no matter how briefly mentioned.
2. Use standard academic terminology for concept names.
3. Distinguish between CORE concepts (main topic being taught) and SUPPORTING concepts 
   (prerequisites or tools used in explanation).
4. Capture the pedagogical context — HOW the teacher explains each concept matters.
5. Note any analogies, examples, or mental models the teacher uses.
6. If the teacher builds one concept on top of another, note that relationship.
"""


def extract_concepts(normalized_data: dict, transcript_data: dict | None = None) -> dict:
    """
    Extract technical concepts from the normalized transcript.

    Args:
        normalized_data: Output from Step 4 (normalized transcript).
        transcript_data: Original transcript data for timing info (optional).

    Returns:
        Dict with extracted concepts and metadata.
    """
    normalized_text = normalized_data.get("normalized_text", "")
    topic_shifts = normalized_data.get("topic_shifts", [])
    term_mappings = normalized_data.get("technical_terms_mapped", {})

    prompt = f"""Analyze this normalized educational transcript and extract ALL technical concepts taught.

Topic shifts detected: {json.dumps(topic_shifts)}
Technical terms found: {json.dumps(list(term_mappings.keys()))}

Transcript:
---
{normalized_text}
---

Return JSON with this EXACT structure:
{{
  "lecture_topic": "The main topic of the lecture",
  "lecture_summary": "A 2-3 sentence summary of what the lecture covers",
  "concepts": [
    {{
      "id": "concept_1",
      "name": "Standard Academic Name",
      "aliases": ["any other names used", "colloquial terms"],
      "description": "What this concept is, as explained in the lecture",
      "type": "core | supporting | prerequisite",
      "explanation_method": "How the teacher explained it (analogy, example, formal definition, etc.)",
      "key_points": ["important detail 1", "important detail 2"],
      "complexity_note": "Any time/space complexity mentioned, if applicable"
    }}
  ],
  "teaching_flow": ["concept_id1 -> concept_id2", "concept_id2 -> concept_id3"],
  "analogies_used": [
    {{
      "concept": "concept_id",
      "analogy": "The analogy or example used by the teacher"
    }}
  ]
}}

Be thorough — extract every concept, even briefly mentioned ones."""

    print(f"[Step 5] Extracting concepts from normalized transcript...")
    result = call_gemini_json(prompt=prompt, system_instruction=SYSTEM_INSTRUCTION, model=_MODEL)

    # Post-process: ensure concept IDs are unique
    seen_ids = set()
    for concept in result.get("concepts", []):
        if concept["id"] in seen_ids:
            concept["id"] = f"{concept['id']}_{len(seen_ids)}"
        seen_ids.add(concept["id"])

    # Save
    output_path = None
    if transcript_data and "audio_file" in transcript_data:
        audio_stem = Path(transcript_data["audio_file"]).stem
        video_dir = get_video_dir(audio_stem)
        output_path = video_dir / f"{audio_stem}_concepts.json"
    else:
        video_dir = get_video_dir("unknown")
        output_path = video_dir / "concepts.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    num_concepts = len(result.get("concepts", []))
    core = sum(1 for c in result.get("concepts", []) if c.get("type") == "core")
    supporting = num_concepts - core

    print(f"[Step 5] Lecture topic: {result.get('lecture_topic', 'Unknown')}")
    print(f"[Step 5] Concepts extracted: {num_concepts} ({core} core, {supporting} supporting)")
    print(f"[Step 5] Saved to: {output_path}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.concept_extract <normalized.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    result = extract_concepts(data)
    print(f"\n--- Concepts ---")
    for c in result.get("concepts", []):
        print(f"  [{c['type']}] {c['name']}: {c['description'][:80]}...")
