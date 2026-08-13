"""Resolve cover artwork for a conversion run.

Three tiers, first hit wins:

1. A Jellyfin sidecar image in the season or show folder.
2. A cover image embedded in the video container (a Matroska attachment,
   which ffmpeg exposes as a video stream).
3. A frame grabbed from the episode itself.

Every tier funnels through the same normalization step, so callers get one
uniform JPEG regardless of source.  Nothing here raises: artwork is a nicety
and must never fail a conversion.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional, Sequence

from tvshow_noisemaker.metadata import _extract_season_number

logger = logging.getLogger(__name__)

# Long edge of the embedded cover, in pixels.
_MAX_EDGE = 600

# Sidecar base names, in preference order, and the extensions tried for each.
_SIDECAR_NAMES = ("poster", "folder", "cover", "default")
_SIDECAR_EXTS = (".jpg", ".jpeg", ".png")

# Image codecs that can appear as an attached cover inside a video container.
_COVER_CODECS = {"mjpeg", "png"}

# Seconds past the detected intro to seek when grabbing a fallback frame.
_FRAME_GRAB_OFFSET = 60.0

_FFMPEG_TIMEOUT = 60


def _first_existing(directory: Path, names: Sequence[str]) -> Optional[Path]:
    """Return the first ``<name><ext>`` file that exists in *directory*."""
    for name in names:
        for ext in _SIDECAR_EXTS:
            candidate = directory / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return None


def _find_sidecar(input_dir: str) -> Optional[Path]:
    """Find a Jellyfin sidecar image for *input_dir*, searching outward."""
    season_dir = Path(input_dir)

    found = _first_existing(season_dir, _SIDECAR_NAMES)
    if found:
        return found

    # Only look at the parent when this really is a season folder.  Otherwise
    # the parent is the library root, whose artwork belongs to another show.
    season = _extract_season_number(season_dir.name)
    show_dir = season_dir.parent
    if season is None or show_dir == season_dir:
        return None

    found = _first_existing(show_dir, (f"season{season:02d}-poster",))
    if found:
        return found
    return _first_existing(show_dir, _SIDECAR_NAMES)


def _find_embedded_cover_index(video_path: str) -> Optional[int]:
    """Return the stream index of a cover image inside *video_path*.

    Matroska attachments surface as video streams, so the primary video track
    has to be told apart from the cover.  On the observed library the cover
    carries ``filename``/``mimetype`` tags and no ``attached_pic`` disposition,
    and its index varies between files -- hence the probe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries",
        "stream=index,codec_name:stream_disposition=attached_pic:stream_tags=filename,mimetype",
        "-of", "json",
        video_path,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Artwork: ffprobe failed on %s: %s", video_path, exc)
        return None

    if result.returncode != 0:
        logger.warning(
            "Artwork: ffprobe exited %d on %s", result.returncode, video_path
        )
        return None

    try:
        streams = json.loads(result.stdout).get("streams", [])
    except (ValueError, AttributeError) as exc:
        logger.warning("Artwork: could not parse ffprobe output: %s", exc)
        return None

    # ffprobe emits streams in index order, so the first match is the lowest.
    for stream in streams:
        if stream.get("codec_name") not in _COVER_CODECS:
            continue
        tags = stream.get("tags") or {}
        disposition = stream.get("disposition") or {}
        if tags.get("filename") or tags.get("mimetype") or disposition.get("attached_pic"):
            return stream.get("index")
    return None
