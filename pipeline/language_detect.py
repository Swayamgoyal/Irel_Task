"""
Step 3: Language Detection & Code-Mix Analysis

Uses Unicode script detection for word-level language tagging —
this is more effective for code-mixed text than sentence-level classifiers
like IndicLID, because languages switch mid-sentence.

Identifies:
  - Script of each word (Devanagari, Tamil, Latin, etc.)
  - Per-segment language mix ratios
  - Overall language profile of the transcript

Inspired by AI4Bharat's IndicLID approach but optimized for code-mixed input.
"""
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field, asdict

# Unicode block ranges for Indian scripts
SCRIPT_RANGES = {
    "Devanagari":  (0x0900, 0x097F),   # Hindi, Marathi, Sanskrit
    "Bengali":     (0x0980, 0x09FF),   # Bengali, Assamese
    "Gurmukhi":    (0x0A00, 0x0A7F),   # Punjabi
    "Gujarati":    (0x0A80, 0x0AFF),   # Gujarati
    "Oriya":       (0x0B00, 0x0B7F),   # Odia
    "Tamil":       (0x0B80, 0x0BFF),   # Tamil
    "Telugu":      (0x0C00, 0x0C7F),   # Telugu
    "Kannada":     (0x0C80, 0x0CFF),   # Kannada
    "Malayalam":   (0x0D00, 0x0D7F),   # Malayalam
    "Latin":       (0x0000, 0x024F),   # English and romanized text
}

# Map scripts to likely languages (for code-mixed contexts)
SCRIPT_TO_LANGUAGE = {
    "Devanagari": "Hindi",
    "Bengali": "Bengali",
    "Gurmukhi": "Punjabi",
    "Gujarati": "Gujarati",
    "Oriya": "Odia",
    "Tamil": "Tamil",
    "Telugu": "Telugu",
    "Kannada": "Kannada",
    "Malayalam": "Malayalam",
    "Latin": "English",
}


@dataclass
class WordTag:
    """A word with its detected script and language."""
    word: str
    script: str
    language: str


@dataclass
class SegmentLanguageProfile:
    """Language profile for a transcript segment."""
    segment_id: int
    start: float
    end: float
    text: str
    words: list[WordTag] = field(default_factory=list)
    language_distribution: dict[str, float] = field(default_factory=dict)
    primary_language: str = ""
    is_code_mixed: bool = False


@dataclass
class TranscriptLanguageProfile:
    """Language profile for the entire transcript."""
    languages_detected: list[str] = field(default_factory=list)
    overall_distribution: dict[str, float] = field(default_factory=dict)
    code_mix_ratio: float = 0.0  # 0 = monolingual, 1 = heavily mixed
    segment_profiles: list[SegmentLanguageProfile] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_script(char: str) -> str:
    """Detect the Unicode script of a character."""
    code = ord(char)
    for script, (start, end) in SCRIPT_RANGES.items():
        if start <= code <= end:
            # For Latin, only count actual letters, not punctuation/numbers
            if script == "Latin" and not char.isalpha():
                continue
            return script
    return "Unknown"


def detect_word_language(word: str) -> tuple[str, str]:
    """
    Detect the script and language of a word based on its characters.

    Returns:
        (script, language) tuple.
    """
    script_counts = Counter()
    for char in word:
        if char.isalpha():
            script = detect_script(char)
            if script != "Unknown":
                script_counts[script] += 1

    if not script_counts:
        return "Unknown", "Unknown"

    dominant_script = script_counts.most_common(1)[0][0]
    language = SCRIPT_TO_LANGUAGE.get(dominant_script, "Unknown")
    return dominant_script, language


def analyze_segment(segment_id: int, start: float, end: float, text: str) -> SegmentLanguageProfile:
    """
    Analyze a single transcript segment for language mixing.

    Performs word-level language tagging and computes language distribution.
    """
    # Split into words (keep Indic + Latin characters)
    words_raw = re.findall(r'[\w\u0900-\u0D7F]+', text)

    word_tags = []
    lang_counts = Counter()

    for w in words_raw:
        script, language = detect_word_language(w)
        if language != "Unknown":
            word_tags.append(WordTag(word=w, script=script, language=language))
            lang_counts[language] += 1

    total = sum(lang_counts.values()) or 1
    distribution = {lang: round(count / total, 3) for lang, count in lang_counts.most_common()}

    primary = lang_counts.most_common(1)[0][0] if lang_counts else "Unknown"
    is_mixed = len([l for l, c in lang_counts.items() if c / total > 0.1]) > 1

    return SegmentLanguageProfile(
        segment_id=segment_id,
        start=start,
        end=end,
        text=text,
        words=word_tags,
        language_distribution=distribution,
        primary_language=primary,
        is_code_mixed=is_mixed,
    )


def analyze_transcript(transcript_data: dict) -> TranscriptLanguageProfile:
    """
    Analyze full transcript for language mixing patterns.

    Args:
        transcript_data: Parsed transcript JSON from Step 2.

    Returns:
        TranscriptLanguageProfile with per-segment and overall analysis.
    """
    segments = transcript_data.get("segments", [])
    segment_profiles = []
    overall_lang_counts = Counter()

    for seg in segments:
        profile = analyze_segment(
            segment_id=seg["id"],
            start=seg["start"],
            end=seg["end"],
            text=seg["text"],
        )
        segment_profiles.append(profile)

        for tag in profile.words:
            overall_lang_counts[tag.language] += 1

    total = sum(overall_lang_counts.values()) or 1
    overall_dist = {lang: round(count / total, 3) for lang, count in overall_lang_counts.most_common()}

    languages = list(overall_dist.keys())

    # Code-mix ratio: 1 - (dominant language share)
    dominant_share = max(overall_dist.values()) if overall_dist else 1.0
    code_mix_ratio = round(1.0 - dominant_share, 3)

    mixed_segments = sum(1 for sp in segment_profiles if sp.is_code_mixed)

    print(f"[Step 3] Languages detected: {languages}")
    print(f"[Step 3] Distribution: {overall_dist}")
    print(f"[Step 3] Code-mix ratio: {code_mix_ratio:.1%}")
    print(f"[Step 3] Mixed segments: {mixed_segments}/{len(segment_profiles)}")

    return TranscriptLanguageProfile(
        languages_detected=languages,
        overall_distribution=overall_dist,
        code_mix_ratio=code_mix_ratio,
        segment_profiles=segment_profiles,
    )


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.language_detect <transcript.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    profile = analyze_transcript(data)

    # Print some mixed segments as examples
    mixed = [sp for sp in profile.segment_profiles if sp.is_code_mixed][:5]
    print(f"\n--- Example code-mixed segments ---")
    for sp in mixed:
        print(f"  [{sp.start:.1f}s] {sp.text[:100]}...")
        print(f"    Languages: {sp.language_distribution}")
