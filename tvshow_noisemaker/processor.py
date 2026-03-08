"""Per-file processing: trim audio and convert to MP3.

Uses ``ffmpeg`` (via ``subprocess``) to:
1. Seek to *intro_end* (start of content).
2. Read until *outro_start* (end of content).
3. Encode the extracted audio as MP3 at the requested bitrate.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def process_file(
    input_path: str,
    output_path: str,
    intro_end: float,
    outro_start: float,
    mp3_bitrate: int = 192,
) -> None:
    """Trim *input_path* and write an MP3 to *output_path*.

    Parameters
    ----------
    input_path:
        Absolute path to the source video file.
    output_path:
        Absolute path for the destination ``.mp3`` file.
    intro_end:
        Start time (seconds) of the content we want to keep.
    outro_start:
        End time (seconds) of the content we want to keep.
    mp3_bitrate:
        MP3 encoding bitrate in kbps (e.g. 192).

    Raises
    ------
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    ValueError
        If *intro_end* >= *outro_start* (nothing to encode).
    """
    if outro_start <= intro_end:
        raise ValueError(
            f"outro_start ({outro_start:.2f}s) must be greater than "
            f"intro_end ({intro_end:.2f}s) for {input_path}"
        )

    duration = outro_start - intro_end
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",                          # overwrite without asking
        "-ss", str(intro_end),         # seek to content start (fast seek)
        "-t", str(duration),           # duration to encode
        "-i", input_path,
        "-vn",                         # drop video stream
        "-acodec", "libmp3lame",
        "-ab", f"{mp3_bitrate}k",
        "-ac", "2",                    # stereo output
        output_path,
    ]

    logger.debug("ffmpeg cmd: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out processing {input_path}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable not found – please install ffmpeg and ensure it is on PATH"
        ) from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {input_path} (exit {result.returncode}):\n"
            f"{result.stderr[-2000:]}"
        )

    logger.info("Written: %s", output_path)


def build_output_path(input_path: str, output_dir: str) -> str:
    """Derive the output ``.mp3`` path from *input_path* and *output_dir*."""
    stem = Path(input_path).stem
    return str(Path(output_dir) / f"{stem}.mp3")
