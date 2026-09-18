"""Per-file processing: trim audio and convert to MP3.

Uses ``ffmpeg`` (via ``subprocess``) to either:
* **Simple trim** – seek to *intro_end*, read until *outro_start*.
* **Concat trim** – when there is a cold open before the intro/theme,
  keep ``[0 .. intro_start]`` and ``[intro_end .. outro_start]`` and
  join them with the ffmpeg ``concat`` audio filter.
"""

from __future__ import annotations

import functools
import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum pre-intro content (seconds) to bother preserving via concat.
_MIN_COLD_OPEN = 1.0

# format name -> (extension, codec args, uses_bitrate)
_FORMATS = {
    "mp3": (".mp3", ["-acodec", "libmp3lame"], True),
    "m4a": (".m4a", ["-acodec", "aac"], True),
    "flac": (".flac", ["-acodec", "flac"], False),
    "wav": (".wav", ["-acodec", "pcm_s16le"], False),
}


@functools.lru_cache(maxsize=None)
def _aac_encoder() -> str:
    """``aac_at`` (Apple AudioToolbox, several times faster than ffmpeg's ``aac``) when this ffmpeg has it."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=10,
        )
    except Exception:  # pylint: disable=broad-except
        return "aac"
    if result.returncode != 0:
        return "aac"
    return "aac_at" if " aac_at " in result.stdout else "aac"


def _codec_args(output_format: str, mp3_bitrate: int) -> list[str]:
    _, codec_args, uses_bitrate = _FORMATS[output_format]
    args = list(codec_args)
    if output_format == "m4a":
        args = ["-acodec", _aac_encoder()]
    if uses_bitrate:
        args.extend(["-ab", f"{mp3_bitrate}k"])
    return args


def process_file(
    input_path: str,
    output_path: str,
    intro_end: float,
    outro_start: float,
    intro_start: float = 0.0,
    mp3_bitrate: int = 256,
    output_format: str = "mp3",
    metadata: Optional[dict[str, str]] = None,
    artwork_path: Optional[str] = None,
) -> None:
    """Trim *input_path* and write an MP3 to *output_path*.

    Parameters
    ----------
    input_path:
        Absolute path to the source video file.
    output_path:
        Absolute path for the destination ``.mp3`` file.
    intro_end:
        Time (seconds) where the intro/theme ends (content resumes).
    outro_start:
        Time (seconds) where the outro/credits begin.
    intro_start:
        Time (seconds) where the intro/theme begins.  Content before
        this (the cold open) is preserved.  When 0 the behaviour is
        the same as a simple seek to *intro_end*.
    mp3_bitrate:
        Encoding bitrate in kbps (e.g. 256) for lossy formats (mp3, m4a).
        Ignored for lossless formats (flac, wav).
    output_format:
        One of "mp3", "m4a", "flac", "wav".
    metadata:
        Optional output tags to write into the MP3 with ffmpeg.
    artwork_path:
        Optional path to a cover image to embed.  Ignored for wav output,
        whose muxer rejects video streams.

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

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    has_cold_open = intro_start >= _MIN_COLD_OPEN and intro_start < intro_end

    if has_cold_open:
        cmd = _build_concat_cmd(
            input_path, output_path,
            intro_start, intro_end, outro_start,
            mp3_bitrate, output_format, metadata, artwork_path,
        )
    else:
        cmd = _build_simple_cmd(
            input_path, output_path,
            intro_end, outro_start,
            mp3_bitrate, output_format, metadata, artwork_path,
        )

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


def _build_simple_cmd(
    input_path: str,
    output_path: str,
    intro_end: float,
    outro_start: float,
    mp3_bitrate: int,
    output_format: str = "mp3",
    metadata: Optional[dict[str, str]] = None,
    artwork_path: Optional[str] = None,
) -> list:
    """Build an ffmpeg command that seeks past the intro and encodes to outro_start."""
    duration = outro_start - intro_end
    art_in, art_out = _build_artwork_args(artwork_path, output_format)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(intro_end),
        "-t", str(duration),
        "-i", input_path,
        *art_in,
        "-map", "0:a:0",
        *_codec_args(output_format, mp3_bitrate),
        "-ac", "2",
        *art_out,
    ]
    cmd.extend(_build_metadata_args(metadata, output_format))
    cmd.append(output_path)
    return cmd


def _build_concat_cmd(
    input_path: str,
    output_path: str,
    intro_start: float,
    intro_end: float,
    outro_start: float,
    mp3_bitrate: int,
    output_format: str = "mp3",
    metadata: Optional[dict[str, str]] = None,
    artwork_path: Optional[str] = None,
) -> list:
    """Build an ffmpeg command that keeps [0..intro_start] + [intro_end..outro_start]."""
    filter_complex = (
        f"[0:a]atrim=start=0:end={intro_start},asetpts=PTS-STARTPTS[pre];"
        f"[0:a]atrim=start={intro_end}:end={outro_start},asetpts=PTS-STARTPTS[post];"
        f"[pre][post]concat=n=2:v=0:a=1[out]"
    )
    art_in, art_out = _build_artwork_args(artwork_path, output_format)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        *art_in,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        *_codec_args(output_format, mp3_bitrate),
        "-ac", "2",
        *art_out,
    ]
    cmd.extend(_build_metadata_args(metadata, output_format))
    cmd.append(output_path)
    return cmd


def _build_metadata_args(
    metadata: Optional[dict[str, str]], output_format: str = "mp3"
) -> list[str]:
    if not metadata:
        return []

    args = ["-id3v2_version", "3"] if output_format == "mp3" else []
    for key in ("artist", "album", "title", "track", "genre"):
        value = metadata.get(key)
        if value:
            args.extend(["-metadata", f"{key}={value}"])
    return args


def _build_artwork_args(
    artwork_path: Optional[str], output_format: str
) -> tuple[list[str], list[str]]:
    """Return ``(input_args, output_args)`` for embedding cover art.

    WAV gets nothing: the wav muxer rejects video streams outright, so mapping
    an image into it fails the encode rather than being ignored.
    """
    if not artwork_path or output_format == "wav":
        return [], []
    return (
        ["-i", artwork_path],
        ["-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic"],
    )


def build_output_path(input_path: str, output_dir: str, output_format: str = "mp3") -> str:
    """Derive the output path from *input_path* and *output_dir* for *output_format*."""
    stem = Path(input_path).stem
    ext, _, _ = _FORMATS[output_format]
    return str(Path(output_dir) / f"{stem}{ext}")
