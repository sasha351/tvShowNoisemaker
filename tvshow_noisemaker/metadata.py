"""Infer MP3 metadata from TV episode paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

_EPISODE_PATTERNS = [
    re.compile(r"(?i)\bs(?P<season>\d{1,2})[ ._-]*e(?P<episode>\d{1,3})\b"),
    re.compile(r"(?i)\b(?P<season>\d{1,2})x(?P<episode>\d{1,3})\b"),
    re.compile(r"(?i)\bseason[ ._-]*(?P<season>\d{1,2})[ ._-]*episode[ ._-]*(?P<episode>\d{1,3})\b"),
    re.compile(r"(?i)\bepisode[ ._-]*(?P<episode>\d{1,3})\b"),
]
_SEASON_ONLY_PATTERNS = [
    re.compile(r"(?i)\bseason[ ._-]*(?P<season>\d{1,2})\b"),
    re.compile(r"(?i)\bs(?P<season>\d{1,2})\b"),
]
_IGNORED_TITLE_TOKENS = {
    "1080p",
    "720p",
    "480p",
    "2160p",
    "web",
    "webrip",
    "webdl",
    "web-dl",
    "bluray",
    "brrip",
    "dvdrip",
    "hdrip",
    "hdtv",
    "x264",
    "x265",
    "h264",
    "h265",
    "hevc",
    "aac",
    "ac3",
    "ddp5",
    "ddp51",
    "10bit",
    "proper",
    "repack",
    "rerip",
    "remux",
    "amzn",
    "dsnp",
    "nf",
    "dl",
}


def infer_track_metadata(input_path: str, input_dir: str) -> dict[str, str]:
    """Infer artist/album/title/track/genre metadata for an episode."""
    episode_path = Path(input_path)
    stem = episode_path.stem
    show_name = infer_show_name(input_dir)
    season_number = _extract_season_number(stem) or _extract_season_number(Path(input_dir).name)
    episode_number = _extract_episode_number(stem)
    episode_title = _extract_episode_title(stem, show_name)

    metadata = {
        "artist": show_name,
        "genre": "TV Show",
    }

    if season_number is not None:
        metadata["album"] = f"Season {season_number}"

    title = _build_track_title(episode_number, episode_title)
    if title:
        metadata["title"] = title

    if episode_number is not None:
        metadata["track"] = str(episode_number)

    return metadata


def infer_show_name(input_dir: str) -> str:
    """Infer the series name from the input directory.

    If the input directory itself looks like a season folder, use its parent as
    the show name instead.
    """
    input_path = Path(input_dir)
    folder_name = input_path.name or input_path.parent.name
    if _extract_season_number(folder_name) is not None and input_path.parent != input_path:
        parent_name = _normalize_label(input_path.parent.name)
        if parent_name:
            return parent_name
    return _normalize_label(folder_name)


def _build_track_title(
    episode_number: Optional[int],
    episode_title: Optional[str],
) -> Optional[str]:
    if episode_number is not None and episode_title:
        return f"Episode {episode_number:02d} - {episode_title}"
    if episode_title:
        return episode_title
    if episode_number is not None:
        return f"Episode {episode_number:02d}"
    return None


def _extract_episode_number(name: str) -> Optional[int]:
    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(name)
        if match and match.groupdict().get("episode"):
            return int(match.group("episode"))
    return None


def _extract_season_number(name: str) -> Optional[int]:
    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(name)
        if match and match.groupdict().get("season"):
            return int(match.group("season"))
    for pattern in _SEASON_ONLY_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group("season"))
    return None


def _extract_episode_title(name: str, show_name: Optional[str] = None) -> Optional[str]:
    show_name_norm = _normalize_label(show_name).casefold() if show_name else None
    matched_episode_pattern = False

    for pattern in _EPISODE_PATTERNS:
        match = pattern.search(name)
        if not match:
            continue
        matched_episode_pattern = True
        trailing = _clean_title_fragment(name[match.end():])
        if trailing:
            return trailing
        leading = _clean_title_fragment(name[:match.start()])
        if leading and (show_name_norm is None or leading.casefold() != show_name_norm):
            return leading

    if matched_episode_pattern:
        return None
    return _clean_title_fragment(name)


def _clean_title_fragment(fragment: str) -> Optional[str]:
    normalized = fragment.replace("[", " ").replace("]", " ")
    tokens = re.split(r"[\s._-]+", normalized)
    cleaned_tokens: list[str] = []
    for token in tokens:
        if not token:
            continue
        lowered = token.lower()
        if lowered in _IGNORED_TITLE_TOKENS:
            continue
        if re.fullmatch(r"\d{3,4}p", lowered):
            continue
        if re.fullmatch(r"s\d{1,2}e\d{1,3}", lowered):
            continue
        cleaned_tokens.append(token)
    title = " ".join(cleaned_tokens).strip()
    return _normalize_label(title) if title else None


def _normalize_label(value: str) -> str:
    label = re.sub(r"[._-]+", " ", value).strip()
    label = re.sub(r"\s+", " ", label)
    if not label:
        return "Unknown Show"
    if label.islower() or label.isupper():
        return label.title()
    return label