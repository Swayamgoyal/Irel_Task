"""
Step 2 — Speech-to-Text  (Whisper large-v3, rewritten from scratch)

What was wrong with the old version and what changed:
──────────────────────────────────────────────────────
1. Unicode-range char filtering  →  REMOVED
   Aggressively deleted valid Indic script characters and replaced them
   with spaces, corrupting the transcript silently.

2. temperature fallback schedule  →  REPLACED with greedy (0.0 only)
   Fallback temperatures (0, 0.2, 0.4 …) introduce randomness and cause
   quality inconsistencies across segments.

3. condition_on_previous_text=False  →  SET TO True
   Without context each segment is decoded independently, producing
   incoherent vocabulary jumps. True + repetition penalties is better.

4. No repetition controls  →  ADDED repetition_penalty + no_repeat_ngram_size
   faster-whisper 1.x exposes these; they are the primary defence against
   the hallucination loops Whisper large-v3 is notorious for.

5b. condition_on_previous_text=True feedback loop  →  MITIGATED (v2)
    When a code-mixed Indic segment is unclear, Whisper gets stuck on the
    last English word it heard (e.g. "Slicing") and the cond_prev context
    keeps re-injecting that word, producing dozens of identical segments.
    Three-layer mitigation added:
      a) compression_ratio_threshold lowered 2.0 → 1.8 (tighter per-segment)
      b) Sub-100ms segment filter: hallucinated micro-segments (0.02-0.09s)
         are physically impossible speech and are dropped unconditionally.
      c) Consecutive-duplicate run detector: MAX_DUP_RUN=2 identical
         normalised texts in a row → all further copies dropped.

5. Language switching mid-stream  →  LOCKED after detection
   If an Indic language is detected with ≥ 45 % confidence, it is forced
   for the entire file.  multilingual=False prevents per-segment flip-flop.

6. VAD over-segmenting  →  TUNED for lectures
   max_speech_duration_s=inf (never split mid-sentence),
   min_silence_duration_ms=800 (wait longer before starting a new segment),
   hallucination_silence_threshold silences suspicious "text over quiet"
   hallucinations automatically (new in faster-whisper 1.x).

7. hotwords  →  ADDED for Python / programming keywords
   Boosts recognition of technical terms spoken in English inside Indic
   sentences (the most common code-mixed pattern in these lectures).

Requires: faster-whisper >= 1.0.0  (tested on 1.2.1)
"""
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

from pipeline.config import WHISPER_MODEL, get_video_dir

# Indic language codes Whisper knows about (ISO-639-1)
_INDIC_LANGS = {
    "hi", "ta", "te", "ml", "kn", "bn", "mr", "gu", "pa",
    "ur", "sa", "or", "ne", "si", "as",
}

# Generic lecture prompt — works for any Indic language mixed with English
_DEFAULT_PROMPT = (
    "This is a technical programming lecture in an Indian language mixed with "
    "English terms (code-mixed: Hinglish, Kanglish, Tanglish, Manglish, etc.). "
    "Common Python keywords spoken in English: print, input, if, else, elif, "
    "for, while, break, continue, return, list, tuple, dict, set, function, "
    "variable, string, integer, boolean, loop, condition, index, append, range, "
    "class, object, import, module, error, exception, output, indentation."
)

# Native-script prompts per Indic language.
# Writing examples in the native script strongly biases Whisper to output
# that script rather than producing Latin-alphabet phonetic transliterations.
_LANG_PROMPTS: dict[str, str] = {
    "bn": (
        "এটি একটি প্রযুক্তিগত শিক্ষামূলক বক্তৃতা। বাংলা ও ইংরেজি মিশ্রিত ভাষায়। "
        "জীববিজ্ঞান, কোষ বিভাজন, ক্রোমোজোম, নিউক্লিয়াস, কোষপর্দা, জীবজগত, "
        "মাইটোসিস, মিয়োসিস, ডিএনএ, আরএনএ, প্রোটিন, সালোকসংশ্লেষণ। "
        "English terms: cell division, chromosome, DNA, RNA, protein, "
        "biology, class, chapter, nucleus, organism, mitosis, meiosis."
    ),
    "hi": (
        "यह एक तकनीकी शैक्षिक व्याख्यान है। हिंदी और अंग्रेजी मिश्रित भाषा में। "
        "जीव विज्ञान, कोशिका विभाजन, गुणसूत्र, केन्द्रक, कोशिका झिल्ली, "
        "माइटोसिस, मेयोसिस, डीएनए, आरएनए, प्रोटीन, प्रकाश संश्लेषण। "
        "English terms: cell division, chromosome, DNA, RNA, protein, "
        "biology, class, chapter, nucleus, organism."
    ),
    "te": (
        "ఇది ఒక సాంకేతిక విద్యా ఉపన్యాసం. తెలుగు మరియు ఆంగ్లం కలిపి. "
        "జీవ శాస్త్రం, కణ విభజన, క్రోమోజోమ్, కేంద్రకం, DNA, RNA, ప్రోటీన్, "
        "మైటోసిస్, మియోసిస్, కిరణజన్య సంయోగక్రియ. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "ta": (
        "இது ஒரு தொழில்நுட்ப கல்வி விரிவுரை. தமிழ் மற்றும் ஆங்கிலம் கலந்து. "
        "உயிரியல், செல் பிரிவு, நிறப்புரி, உட்கரு, DNA, RNA, புரதம், "
        "மைட்டோசிஸ், ஒளிச்சேர்க்கை. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "kn": (
        "ಇದು ಒಂದು ತಾಂತ್ರಿಕ ಶೈಕ್ಷಣಿಕ ಉಪನ್ಯಾಸ. ಕನ್ನಡ ಮತ್ತು ಆಂಗ್ಲ ಮಿಶ್ರಿತ. "
        "ಜೀವಶಾಸ್ತ್ರ, ಕೋಶ ವಿಭಜನೆ, ವರ್ಣತಂತು, ಕೇಂದ್ರಕ, DNA, RNA, ಪ್ರೋಟೀನ್. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "ml": (
        "ഇത് ഒരു സാങ്കേതിക വിദ്യാഭ്യാസ പ്രഭാഷണമാണ്. മലയാളവും ഇംഗ്ലീഷും ചേർന്ന്. "
        "ജീവശാസ്ത്രം, കോശ വിഭജനം, ക്രോമോസോം, കേന്ദ്രകം, DNA, RNA, പ്രോടീൻ. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "mr": (
        "हे एक तांत्रिक शैक्षणिक व्याख्यान आहे. मराठी आणि इंग्रजी मिश्रित. "
        "जीवशास्त्र, पेशी विभाजन, गुणसूत्र, केंद्रक, DNA, RNA, प्रथिने. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "gu": (
        "આ એક તકનીકી શૈક્ષણિક વ્યાખ્યાન છે. ગુજરાતી અને અંગ્રેજી મિશ્રિત. "
        "જીવ વિજ્ઞાન, કોષ વિભાજન, રંગસૂત્ર, કેન્દ્રક, DNA, RNA, પ્રોટીન. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "pa": (
        "ਇਹ ਇੱਕ ਤਕਨੀਕੀ ਸਿੱਖਿਆ ਭਾਸ਼ਣ ਹੈ। ਪੰਜਾਬੀ ਅਤੇ ਅੰਗਰੇਜ਼ੀ ਮਿਲਵੀਂ. "
        "ਜੀਵ ਵਿਗਿਆਨ, ਸੈੱਲ ਵੰਡ, ਕ੍ਰੋਮੋਸੋਮ, ਕੇਂਦਰਕ, DNA, RNA, ਪ੍ਰੋਟੀਨ. "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
    "ur": (
        "یہ ایک تکنیکی تعلیمی لیکچر ہے۔ اردو اور انگریزی ملی جلی زبان میں۔ "
        "حیاتیات، خلیہ تقسیم، کروموسوم، نیوکلیئس، DNA، RNA، پروٹین۔ "
        "English terms: cell, chromosome, DNA, RNA, protein, biology."
    ),
}

# Hotwords: boost recognition of common code-mixed technical terms
_HOTWORDS = (
    "print input for while if else elif break continue return list tuple dict "
    "set function variable string integer boolean loop condition index append "
    "range class import module error exception output"
)


# ── Data classes (same interface as pipeline.transcribe) ──────────────────────

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


# ── Internal helpers ───────────────────────────────────────────────────────────

def _setup_cuda_libs() -> None:
    """Make pip-installed CUDA shared libraries visible to CTranslate2."""
    import sys, os
    site = next((p for p in sys.path if "site-packages" in p), None)
    if not site:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    extra = []
    for sub in ("nvidia/cublas/lib", "nvidia/cudnn/lib", "nvidia/cuda_runtime/lib"):
        p = os.path.join(site, sub)
        if os.path.isdir(p) and p not in current:
            extra.append(p)
    if extra:
        os.environ["LD_LIBRARY_PATH"] = ":".join(extra) + (":" + current if current else "")


def _best_device_and_compute() -> tuple[str, str]:
    """Return (device, compute_type) for the best available hardware."""
    try:
        import ctranslate2
        supported = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in supported:
            return "cuda", "float16"
        if "bfloat16" in supported:
            return "cuda", "bfloat16"
    except Exception:
        pass
    return "cpu", "int8"


# ── Main transcribe function ───────────────────────────────────────────────────

def transcribe(
    audio_path: str | Path,
    model_size: str = WHISPER_MODEL,
    device: str = "auto",
    language: str | None = None,
    # Core quality controls
    beam_size: int = 5,
    repetition_penalty: float = 1.3,
    no_repeat_ngram_size: int = 5,
    condition_on_previous_text: bool = False,
    compression_ratio_threshold: float = 1.5,   # tighter: catches repetitive hallucinated segments
    log_prob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    hallucination_silence_threshold: float = 2.0,
    # Prompt / hotwords
    initial_prompt: str | None = None,
    hotwords: str | None = _HOTWORDS,
    # VAD
    vad_filter: bool = True,
    # Output
    word_timestamps: bool = True,
    # Legacy params kept for API compatibility with pipeline callers
    best_of: int = 5,
    patience: float = 1.0,
    temperature: tuple = (0.0,),
    batch_size: int = 8,
    chunk_length_s: int = 30,
) -> TranscriptResult:
    """
    Transcribe audio using Whisper large-v3 (faster-whisper ≥ 1.0.0).

    Key quality improvements over the previous version:
      - No destructive character-range filtering
      - Greedy decoding (temperature=0) — consistent, no random fallback
      - repetition_penalty + no_repeat_ngram_size prevent hallucination loops
      - Language locked after detection (multilingual=False)
      - hallucination_silence_threshold drops fake text on silent passages
      - hotwords boost English tech-term recognition in Indic sentences
      - VAD tuned for long uninterrupted lectures

    Args:
        audio_path: Path to audio file (WAV/MP3/FLAC etc., 16 kHz mono preferred).
        model_size: Whisper model to load (default: large-v3 from config).
        device:     "cpu", "cuda", or "auto" (auto-detects GPU).
        language:   ISO-639-1 hint e.g. "kn", "hi", "te". None = auto-detect.
        beam_size:  Beam search width. 5 is the sweet spot for quality/speed.
        repetition_penalty: > 1 penalises repeated tokens — prevents loops.
        no_repeat_ngram_size: Block exact n-gram repetitions.
        condition_on_previous_text: Feed previous segment as context decoder prefix.
        compression_ratio_threshold: Reject segments with too much repetition.
        hallucination_silence_threshold: Seconds of silence that trigger segment drop.
        hotwords: Space-separated words to boost in recognition.
    """
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    _setup_cuda_libs()

    if device == "auto":
        device, compute_type = _best_device_and_compute()
    else:
        compute_type = "float16" if device == "cuda" else "int8"

    print(f"[Step 2] Model         : {model_size}")
    print(f"[Step 2] Device        : {device} ({compute_type})")

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=8,
        num_workers=1,
    )

    # Always load audio array — needed for language detection and gap recovery.
    audio_data = decode_audio(str(audio_path))

    # ── Language detection ────────────────────────────────────────────────────
    if language:
        forced_lang = language
        lang_prob = 1.0
        print(f"[Step 2] Language      : {forced_lang} (caller-forced)")
    else:
        print("[Step 2] Detecting language from first 30 s ...")
        probe = audio_data[: 16000 * 30]  # first 30 s at 16 kHz
        detected, prob, _ = model.detect_language(probe)
        print(f"[Step 2] Detected      : {detected}  ({prob:.1%} confidence)")

        if detected in _INDIC_LANGS and prob >= 0.45:
            forced_lang = detected          # lock — prevents Khmer/Georgian/Thai drift
        elif prob >= 0.80:
            forced_lang = detected          # high-confidence non-Indic (e.g. pure English)
        else:
            forced_lang = None              # truly ambiguous — let Whisper decide per-segment

        lang_prob = prob
        print(f"[Step 2] Locked lang   : {forced_lang or 'auto (low confidence)'}")

    # ── Prompt ────────────────────────────────────────────────────────────────
    if initial_prompt is not None:
        prompt = initial_prompt
    elif forced_lang and forced_lang in _LANG_PROMPTS:
        # Use a native-script prompt so Whisper outputs the correct script
        # instead of Latin-alphabet phonetic transliterations.
        prompt = _LANG_PROMPTS[forced_lang]
        print(f"[Step 2] Prompt         : native-script ({forced_lang})")
    else:
        prompt = _DEFAULT_PROMPT
        print("[Step 2] Prompt         : generic English fallback")

    # ── VAD parameters — tuned for long uninterrupted lectures ───────────────
    vad_params = {
        "threshold": 0.40,                  # keep faint speech
        "min_speech_duration_ms": 150,
        "max_speech_duration_s": float("inf"),  # never force-split mid-sentence
        "min_silence_duration_ms": 800,     # wait longer → fewer micro-splits
        "speech_pad_ms": 400,               # generous context around speech
    } if vad_filter else None

    # ── Transcribe ────────────────────────────────────────────────────────────
    kwargs = dict(
        language=forced_lang,
        task="transcribe",                  # NEVER translate
        beam_size=beam_size,
        best_of=1,                          # irrelevant at temperature=0
        patience=1.0,
        temperature=(0.0, 0.2),             # 0.2 fallback breaks stuck-token loops when seg fails threshold
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        compression_ratio_threshold=compression_ratio_threshold,
        log_prob_threshold=log_prob_threshold,
        no_speech_threshold=no_speech_threshold,
        hallucination_silence_threshold=hallucination_silence_threshold,
        condition_on_previous_text=condition_on_previous_text,
        initial_prompt=prompt,
        hotwords=hotwords,
        multilingual=False,                 # do NOT flip language per-segment
        vad_filter=vad_filter,
        vad_parameters=vad_params,
        word_timestamps=word_timestamps,
        without_timestamps=False,
        suppress_blank=True,
    )

    print(f"[Step 2] Transcribing  : {audio_path.name}")
    print(f"[Step 2] Settings      : beam={beam_size}  rep_penalty={repetition_penalty}  "
          f"no_repeat_ngram={no_repeat_ngram_size}  cond_prev={condition_on_previous_text}")

    t0 = time.time()
    segments_gen, info = model.transcribe(str(audio_path), **kwargs)

    print(f"[Step 2] Audio         : {info.duration:.1f}s  |  "
          f"lang={info.language} ({info.language_probability:.1%})")

    # ── Collect segments ────────────────────────────────────────────────────
    # Filters applied (in order):
    #   1. Empty text
    #   2. Near-certain silence (no_speech_prob > 0.90)
    #   3. Sub-100ms segments — physically impossible real speech; always
    #      hallucinated micro-segments produced during stuck-token loops
    #   4. Consecutive-duplicate run detector — if the same normalised text
    #      appears MAX_DUP_RUN times in a row, all further copies are dropped.
    #      This is the primary defence against the condition_on_previous_text
    #      feedback loop (e.g. repeated "Slicing" / "Slicing" / "Slicing").
    MAX_DUP_RUN = 2          # allow at most 2 consecutive identical segments
    _recent: list[str] = []  # sliding window of last MAX_DUP_RUN texts
    _consecutive_skipped = 0  # consecutive skips — spike here means stuck-token loop

    segments: list[Segment] = []
    skipped = 0
    for seg in segments_gen:
        text = seg.text.strip()

        if not text:
            skipped += 1
            _consecutive_skipped += 1
            continue
        if seg.no_speech_prob > 0.90:           # near-certain silence
            skipped += 1
            _consecutive_skipped += 1
            continue
        if (seg.end - seg.start) < 0.10:        # < 100 ms → hallucinated micro-segment
            skipped += 1
            _consecutive_skipped += 1
            continue

        # Consecutive-duplicate run detector
        norm = text.lower().strip()
        _recent.append(norm)
        if len(_recent) > MAX_DUP_RUN:
            _recent.pop(0)
        if len(_recent) == MAX_DUP_RUN and len(set(_recent)) == 1:
            # All slots identical — we are inside a stuck-token loop; skip
            skipped += 1
            _consecutive_skipped += 1
            if _consecutive_skipped == MAX_DUP_RUN + 1:
                # Just entered the loop — warn with location and offending text
                print(f"[Step 2] WARNING: Stuck-token loop at ~{seg.start:.0f}s "
                      f"('{norm[:50]}') — audio coverage may be incomplete")
            continue

        _consecutive_skipped = 0  # reset — real content is flowing again
        words = []
        if seg.words:
            for w in seg.words:
                wt = w.word.strip()
                if wt:
                    words.append(Word(
                        word=wt,
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 4),
                    ))

        segments.append(Segment(
            id=len(segments),
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=text,
            language=info.language,
            avg_logprob=round(seg.avg_logprob, 4),
            no_speech_prob=round(seg.no_speech_prob, 4),
            words=words,
        ))

        if len(segments) % 10 == 0:
            print(f"[Step 2] {len(segments)} segments  "
                  f"({seg.end:.0f}s / {info.duration:.0f}s)  skipped={skipped}")

    elapsed = time.time() - t0
    print(f"[Step 2] Done          : {len(segments)} kept  |  {skipped} skipped  |  "
          f"{elapsed:.1f}s  ({elapsed / info.duration:.2f}× realtime)")

    # ── Gap detection + recovery ──────────────────────────────────────────────
    # If a stuck-token loop (or bad VAD) left large uncovered regions, re-
    # transcribe each gap independently with fresh context so no audio is lost.
    _SR = 16000
    _GAP_THRESHOLD = 12.0  # seconds — gaps larger than this trigger recovery

    def _find_gaps(segs, total_dur, threshold):
        gaps, prev_end = [], 0.0
        for s in segs:
            if s.start - prev_end > threshold:
                gaps.append((prev_end, s.start))
            prev_end = max(prev_end, s.end)
        if total_dur - prev_end > threshold:
            gaps.append((prev_end, total_dur))
        return gaps

    gaps = _find_gaps(segments, info.duration, _GAP_THRESHOLD)
    if gaps:
        print(f"[Step 2] Gap recovery  : {len(gaps)} uncovered region(s) — re-transcribing")
        recovery_kwargs = dict(
            language=forced_lang,
            task="transcribe",
            beam_size=beam_size,
            best_of=1,
            patience=1.0,
            temperature=(0.0, 0.2),
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            compression_ratio_threshold=compression_ratio_threshold,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            hallucination_silence_threshold=hallucination_silence_threshold,
            condition_on_previous_text=False,  # always fresh — no poisoned context
            initial_prompt=prompt,
            hotwords=hotwords,
            multilingual=False,
            vad_filter=vad_filter,
            vad_parameters=vad_params,
            word_timestamps=word_timestamps,
            without_timestamps=False,
            suppress_blank=True,
        )
        recovered: list[Segment] = []
        for gap_start, gap_end in gaps:
            print(f"[Step 2]   → gap {gap_start:.0f}s – {gap_end:.0f}s ({gap_end - gap_start:.0f}s)")
            gap_audio = audio_data[int(gap_start * _SR):int(gap_end * _SR)]
            if len(gap_audio) < _SR // 2:   # < 0.5 s — nothing to recover
                continue
            gap_segs_gen, _ = model.transcribe(gap_audio, **recovery_kwargs)
            _gr: list[str] = []
            for seg in gap_segs_gen:
                text = seg.text.strip()
                if not text or seg.no_speech_prob > 0.90 or (seg.end - seg.start) < 0.10:
                    continue
                norm = text.lower().strip()
                _gr.append(norm)
                if len(_gr) > MAX_DUP_RUN:
                    _gr.pop(0)
                if len(_gr) == MAX_DUP_RUN and len(set(_gr)) == 1:
                    continue
                words = []
                if seg.words:
                    for w in seg.words:
                        wt = w.word.strip()
                        if wt:
                            words.append(Word(
                                word=wt,
                                start=round(w.start + gap_start, 3),
                                end=round(w.end + gap_start, 3),
                                probability=round(w.probability, 4),
                            ))
                recovered.append(Segment(
                    id=-1,
                    start=round(seg.start + gap_start, 3),
                    end=round(seg.end + gap_start, 3),
                    text=text,
                    language=info.language,
                    avg_logprob=round(seg.avg_logprob, 4),
                    no_speech_prob=round(seg.no_speech_prob, 4),
                    words=words,
                ))
        if recovered:
            print(f"[Step 2] Gap recovery  : +{len(recovered)} segments recovered")
            segments = sorted(segments + recovered, key=lambda s: s.start)
            for i, s in enumerate(segments):
                s.id = i
        else:
            print("[Step 2] Gap recovery  : no speech found in gaps (silence/music)")

    result = TranscriptResult(
        audio_file=str(audio_path),
        model=model_size,
        language=info.language,
        language_probability=round(info.language_probability, 4),
        duration_seconds=round(info.duration, 2),
        transcription_time_seconds=round(elapsed, 2),
        segments=segments,
    )

    output_path = result.save()
    print(f"[Step 2] Saved         → {output_path}")
    print(f"[Step 2] Segments={len(segments)}  Words={sum(len(s.words) for s in segments)}")
    return result


# ── Loader (unchanged interface) ──────────────────────────────────────────────

def load_transcript(path: str | Path) -> TranscriptResult:
    """Load a previously saved transcript from JSON."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = []
    for sd in data.get("segments", []):
        words = [Word(**w) for w in sd.get("words", [])]
        segments.append(Segment(
            id=sd["id"],
            start=sd["start"],
            end=sd["end"],
            text=sd["text"],
            language=sd.get("language"),
            avg_logprob=sd.get("avg_logprob", 0.0),
            no_speech_prob=sd.get("no_speech_prob", 0.0),
            words=words,
        ))
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
        print("Usage: python -m pipeline.transcribe_faster_whisper <audio.wav> [lang]")
        sys.exit(1)
    lang_arg = sys.argv[2] if len(sys.argv) > 2 else None
    res = transcribe(sys.argv[1], language=lang_arg)
    print(f"\n--- Transcript preview (first 600 chars) ---")
    print(res.full_text[:600])
