"""Tests for tvshow_noisemaker.artwork."""

from pathlib import Path

import pytest

from tvshow_noisemaker.artwork import _find_sidecar


def _touch(path: Path) -> Path:
    """Create an empty file, making parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


# ---------------------------------------------------------------------------
# _find_sidecar
# ---------------------------------------------------------------------------


def test_sidecar_prefers_season_folder(tmp_path):
    season = tmp_path / "Family Guy" / "Season 02"
    _touch(season / "poster.jpg")
    _touch(tmp_path / "Family Guy" / "poster.jpg")
    assert _find_sidecar(str(season)) == season / "poster.jpg"


def test_sidecar_falls_back_to_season_numbered_poster(tmp_path):
    """Jellyfin writes per-season art to the show root as seasonNN-poster.jpg."""
    season = tmp_path / "Family Guy" / "Season 02"
    season.mkdir(parents=True)
    _touch(tmp_path / "Family Guy" / "season02-poster.jpg")
    _touch(tmp_path / "Family Guy" / "poster.jpg")
    assert _find_sidecar(str(season)) == tmp_path / "Family Guy" / "season02-poster.jpg"


def test_sidecar_falls_back_to_show_poster(tmp_path):
    season = tmp_path / "Family Guy" / "Season 02"
    season.mkdir(parents=True)
    _touch(tmp_path / "Family Guy" / "folder.jpg")
    assert _find_sidecar(str(season)) == tmp_path / "Family Guy" / "folder.jpg"


def test_sidecar_name_preference_order(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    _touch(season / "cover.jpg")
    _touch(season / "default.jpg")
    _touch(season / "folder.jpg")
    _touch(season / "poster.jpg")
    assert _find_sidecar(str(season)).name == "poster.jpg"


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png"])
def test_sidecar_accepts_each_extension(tmp_path, ext):
    season = tmp_path / "Show" / "Season 01"
    _touch(season / f"poster{ext}")
    assert _find_sidecar(str(season)).name == f"poster{ext}"


def test_sidecar_does_not_escape_a_non_season_folder(tmp_path):
    """A plain folder must not inherit artwork belonging to the library root."""
    show = tmp_path / "Family Guy"
    show.mkdir(parents=True)
    _touch(tmp_path / "poster.jpg")
    assert _find_sidecar(str(show)) is None


def test_sidecar_returns_none_when_absent(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    assert _find_sidecar(str(season)) is None


def test_sidecar_ignores_a_directory_named_like_an_image(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    (season / "poster.jpg").mkdir(parents=True)
    assert _find_sidecar(str(season)) is None
