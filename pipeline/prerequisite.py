"""
Step 6: Prerequisite Dependency Detection

Uses Gemini to detect which concepts are prerequisites for others,
based on the pedagogical flow in the transcript.

This step analyzes:
  - The ORDER in which concepts are introduced
  - Explicit dependency cues ("before we learn X, we need Y")
  - Implicit dependencies (concept X uses concept Y in its explanation)
  - Logical/mathematical dependencies (X requires understanding of Y)
"""
import json
from pathlib import Path

from pipeline.llm_client import call_gemini_json
from pipeline.config import get_video_dir, LLM_MODEL_REASON as _MODEL

SYSTEM_INSTRUCTION = """You are an expert in educational curriculum design and knowledge graph construction.
You specialize in detecting prerequisite relationships between technical concepts.

Your task is to analyze the concepts extracted from an educational lecture and determine 
which concepts must be understood BEFORE others can be learned.

Types of prerequisite relationships:
1. HARD_PREREQUISITE: Concept A MUST be understood before B (e.g., "array" before "sorting")
2. SOFT_PREREQUISITE: Understanding A helps with B but isn't strictly required
3. BUILDS_UPON: Concept B directly extends or refines concept A
4. USES: Concept B uses/applies concept A as a tool

Rules:
1. Base dependencies on the PEDAGOGICAL FLOW — the order the teacher chose matters.
2. Look for explicit cues: "first we need to understand...", "this is similar to..."
3. Look for implicit cues: the teacher explains Y using concepts from X.
4. Do NOT infer dependencies that aren't supported by the transcript content.
5. Every dependency must have a justification from the content.
"""


def detect_prerequisites(concepts_data: dict, normalized_data: dict) -> dict:
    """
    Detect prerequisite dependencies between extracted concepts.

    Args:
        concepts_data: Output from Step 5 (extracted concepts).
        normalized_data: Output from Step 4 (normalized transcript for context).

    Returns:
        Dict with prerequisite relationships.
    """
    concepts = concepts_data.get("concepts", [])
    teaching_flow = concepts_data.get("teaching_flow", [])
    normalized_text = normalized_data.get("normalized_text", "")

    # Build concept summary for the prompt
    concept_list = []
    for c in concepts:
        concept_list.append({
            "id": c["id"],
            "name": c["name"],
            "type": c.get("type", "unknown"),
            "description": c.get("description", ""),
        })

    prompt = f"""Given these concepts extracted from an educational lecture, detect ALL prerequisite dependencies.

Concepts:
{json.dumps(concept_list, indent=2)}

Teaching flow observed: {json.dumps(teaching_flow)}

Full normalized transcript for context:
---
{normalized_text[:6000]}
---

Return JSON with this EXACT structure:
{{
  "dependencies": [
    {{
      "from_concept": "concept_id (prerequisite)",
      "to_concept": "concept_id (dependent)",
      "relationship_type": "HARD_PREREQUISITE | SOFT_PREREQUISITE | BUILDS_UPON | USES",
      "strength": 0.9,
      "justification": "Why this dependency exists based on the lecture content"
    }}
  ],
  "learning_path": ["concept_id1", "concept_id2", "...ordered by recommended learning sequence"],
  "concept_clusters": [
    {{
      "cluster_name": "Name of concept group",
      "concept_ids": ["concept_id1", "concept_id2"],
      "description": "Why these concepts are grouped together"
    }}
  ],
  "root_concepts": ["concept_ids that have no prerequisites (entry points)"],
  "leaf_concepts": ["concept_ids that nothing depends on (end goals)"]
}}

Be thorough and precise. Only include dependencies that are clearly supported by the content."""

    print(f"[Step 6] Detecting prerequisite dependencies among {len(concepts)} concepts...")
    result = call_gemini_json(prompt=prompt, system_instruction=SYSTEM_INSTRUCTION, model=_MODEL)

    # Save
    output_path = get_video_dir("unknown") / "prerequisites.json"
    # Try to derive video_id from normalized_data
    audio_file = ""
    for chunk in normalized_data.get("chunks", []):
        if "original_segments" in chunk:
            break
    # Check if transcript audio_file info is embedded
    stem = Path(normalized_data.get("_audio_file", "unknown")).stem
    if stem != "unknown":
        output_path = get_video_dir(stem) / f"{stem}_prerequisites.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    deps = result.get("dependencies", [])
    hard = sum(1 for d in deps if d.get("relationship_type") == "HARD_PREREQUISITE")
    soft = len(deps) - hard

    print(f"[Step 6] Dependencies found: {len(deps)} ({hard} hard, {soft} soft/other)")
    print(f"[Step 6] Root concepts: {result.get('root_concepts', [])}")
    print(f"[Step 6] Leaf concepts: {result.get('leaf_concepts', [])}")
    print(f"[Step 6] Learning path: {' → '.join(result.get('learning_path', []))}")
    print(f"[Step 6] Saved to: {output_path}")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.prerequisite <concepts.json> <normalized.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        concepts = json.load(f)
    with open(sys.argv[2], "r", encoding="utf-8") as f:
        normalized = json.load(f)

    result = detect_prerequisites(concepts, normalized)
    print(f"\n--- Dependencies ---")
    for dep in result.get("dependencies", []):
        print(f"  {dep['from_concept']} → {dep['to_concept']} "
              f"({dep['relationship_type']}, strength: {dep.get('strength', '?')})")
