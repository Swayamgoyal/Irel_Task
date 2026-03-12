"""
Step 1: Audio Extraction

Handles two input modes:
  1. Local video file  → extract audio with ffmpeg
  2. YouTube URL       → download with yt-dlp, then extract audio

Output: 16 kHz mono WAV file ready for Whisper.
"""
import subprocess
import shutil
from pathlib import Path

from pipeline.config import AUDIO_SAMPLE_RATE, AUDIO_FORMAT, get_video_dir


def _check_dependency(name: str) -> str:
    """Return the path of an executable or raise."""
    path = shutil.which(name)
    if path is None:
        raise EnvironmentError(
            f"'{name}' not found on PATH. Install it first:\n"
            f"  sudo apt install {name}" if name == "ffmpeg" else f"  pip install {name}"
        )
    return path


def _extract_video_id(url: str) -> str:
    """Extract video ID from a YouTube URL."""
    import re
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([\w-]{11})',
        r'([\w-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return 'unknown'


def download_video(url: str, output_dir: Path | None = None) -> Path:
    """
    Download a video from a URL (YouTube, etc.) using yt-dlp.

    Returns the path to the downloaded video file.
    """
    _check_dependency("yt-dlp")
    if output_dir is None:
        video_id = _extract_video_id(url)
        output_dir = get_video_dir(video_id)

    # yt-dlp template: use video id as filename
    output_template = str(output_dir / "%(id)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", output_template,
        "--no-playlist",
        "--restrict-filenames",
        url,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed:\n{result.stderr}")

    # Find the downloaded file from yt-dlp output
    for line in result.stdout.splitlines():
        if "[Merger]" in line or "[download]" in line:
            # Extract destination path
            if "Destination:" in line:
                path_str = line.split("Destination:")[-1].strip()
                p = Path(path_str)
                if p.exists():
                    return p
            if "has already been downloaded" in line:
                path_str = line.split("[download]")[-1].split("has already")[0].strip()
                p = Path(path_str)
                if p.exists():
                    return p
            if "Merging formats into" in line:
                path_str = line.split("Merging formats into")[-1].strip().strip('"')
                p = Path(path_str)
                if p.exists():
                    return p

    # Fallback: find most recently modified mp4 in output dir
    video_files = sorted(output_dir.glob("*.mp4"), key=lambda f: f.stat().st_mtime, reverse=True)
    if video_files:
        return video_files[0]

    raise FileNotFoundError(f"Could not locate downloaded video in {output_dir}")


def extract_audio(
    input_path: str | Path,
    output_path: Path | None = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    loudnorm: bool = False,
) -> Path:
    """
    Extract audio from a video file using ffmpeg.

    Produces a 16 kHz mono WAV suitable for Whisper.

    Args:
        input_path: Path to the video file.
        output_path: Where to save the WAV. Defaults to data/audio/<stem>.wav.
        sample_rate: Target sample rate (default 16000).
        loudnorm: Apply EBU R128 loudness normalisation (-af loudnorm).
                  Recommended for Whisper to improve transcription accuracy.

    Returns:
        Path to the extracted audio file.
    """
    _check_dependency("ffmpeg")
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path is None:
        video_dir = get_video_dir(input_path.stem)
        output_path = video_dir / f"{input_path.stem}.{AUDIO_FORMAT}"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",                        # no video
        "-acodec", "pcm_s16le",       # 16-bit PCM
        "-ar", str(sample_rate),      # resample
        "-ac", "1",                   # mono
    ]
    if loudnorm:
        cmd += ["-af", "loudnorm"]    # EBU R128 loudness normalisation
    cmd += [
        "-y",                         # overwrite
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg produced no output or empty file")

    return output_path


def process_input(source: str, loudnorm: bool = False) -> Path:
    """
    Unified entry point for Step 1.

    Accepts either a local file path or a URL.
    Returns the path to the extracted 16 kHz mono WAV.
    """
    source = source.strip()

    # Determine if source is a URL or local file
    if source.startswith(("http://", "https://", "www.")):
        print(f"[Step 1] Downloading video from: {source}")
        video_path = download_video(source)
        print(f"[Step 1] Downloaded to: {video_path}")
    else:
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

    print(f"[Step 1] Extracting audio from: {video_path}")
    audio_path = extract_audio(video_path, loudnorm=loudnorm)
    print(f"[Step 1] Audio saved to: {audio_path}")
    extra = ", loudnorm" if loudnorm else ""
    print(f"[Step 1] Format: {AUDIO_FORMAT}, {AUDIO_SAMPLE_RATE} Hz, mono{extra}")

    return audio_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.audio_extract <video_path_or_url>")
        sys.exit(1)
    result = process_input(sys.argv[1])
    print(f"\nDone! Audio file: {result}")
