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

import logging
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
