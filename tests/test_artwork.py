"""Tests for tvshow_noisemaker.artwork."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvshow_noisemaker.artwork import _find_embedded_cover_index, _find_sidecar


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


# ---------------------------------------------------------------------------
# _find_embedded_cover_index
# ---------------------------------------------------------------------------


def _ffprobe_result(streams):
    """A MagicMock standing in for a successful ffprobe run."""
    return MagicMock(
        returncode=0, stdout=json.dumps({"streams": streams}), stderr=""
    )


def test_embedded_cover_skips_the_primary_video_stream():
    streams = [
        {"index": 0, "codec_name": "hevc", "disposition": {"attached_pic": 0}},
        {
            "index": 4,
            "codec_name": "mjpeg",
            "disposition": {"attached_pic": 0},
            "tags": {"filename": "cover.jpg", "mimetype": "image/jpeg"},
        },
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") == 4


def test_embedded_cover_index_is_probed_not_assumed():
    """Real files put the cover at varying indices; 3 and 4 both observed."""
    streams = [
        {"index": 0, "codec_name": "hevc", "disposition": {}},
        {"index": 3, "codec_name": "mjpeg", "disposition": {},
         "tags": {"filename": "cover.jpg"}},
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") == 3


def test_embedded_cover_accepts_attached_pic_without_tags():
    streams = [
        {"index": 0, "codec_name": "h264", "disposition": {}},
        {"index": 1, "codec_name": "png", "disposition": {"attached_pic": 1}},
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") == 1


def test_embedded_cover_ignores_untagged_non_cover_image_stream():
    """An mjpeg stream with neither tags nor disposition is not a cover."""
    streams = [
        {"index": 0, "codec_name": "hevc", "disposition": {}},
        {"index": 1, "codec_name": "mjpeg", "disposition": {}},
    ]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_when_no_image_stream():
    streams = [{"index": 0, "codec_name": "hevc", "disposition": {}}]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_on_unparseable_output():
    result = MagicMock(returncode=0, stdout="not json at all", stderr="")
    with patch("subprocess.run", return_value=result):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_on_ffprobe_nonzero_exit():
    result = MagicMock(returncode=1, stdout="", stderr="boom")
    with patch("subprocess.run", return_value=result):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_when_ffprobe_missing():
    with patch("subprocess.run", side_effect=FileNotFoundError("no ffprobe")):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_embedded_cover_none_on_timeout():
    exc = subprocess.TimeoutExpired("ffprobe", 60)
    with patch("subprocess.run", side_effect=exc):
        assert _find_embedded_cover_index("/v.mkv") is None
