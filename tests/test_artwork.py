"""Tests for tvshow_noisemaker.artwork."""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvshow_noisemaker.artwork import (
    _find_embedded_cover_index,
    _find_sidecar,
    _normalize,
    resolve_artwork,
)


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


def test_embedded_cover_probe_requests_disposition_in_combined_entry():
    """Regression guard: disposition must live in the same -show_entries value
    as stream/tags. ffprobe silently drops disposition if it's split into a
    separate -show_entries flag, which mocked-output tests can't detect."""
    streams = [{"index": 0, "codec_name": "hevc", "disposition": {"attached_pic": 0}}]
    with patch("subprocess.run", return_value=_ffprobe_result(streams)) as mock_run:
        _find_embedded_cover_index("/v.mkv")
    cmd = mock_run.call_args.args[0]
    assert (
        "stream=index,codec_name:stream_disposition=attached_pic:stream_tags=filename,mimetype"
        in cmd
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


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------


def test_normalize_builds_a_scale_filter(tmp_path):
    dest = tmp_path / "cover.jpg"

    def _fake_ffmpeg(*args, **kwargs):
        dest.write_bytes(b"jpegdata")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_ffmpeg) as mock_run:
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is True
    cmd = mock_run.call_args[0][0]
    assert cmd[0] == "ffmpeg"
    assert "/src/poster.jpg" in cmd
    assert "scale=w=600:h=600:force_original_aspect_ratio=decrease" in cmd
    assert cmd[-1] == str(dest)


def test_normalize_frame_grab_input_args_stay_ordered_before_i(tmp_path):
    """Regression guard: -ss must precede -i (input seek, not output seek).

    An output seek (-ss after -i) still runs but decodes from frame zero,
    which is far slower -- a correctness-preserving but performance-killing
    reorder that membership-only assertions can't catch.
    """
    dest = tmp_path / "cover.jpg"
    input_args = ["-ss", "90.0", "-i", "/v.mkv", "-map", "0:v:0"]
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        dest.write_bytes(b"jpegdata")
        _normalize(input_args, str(dest))
    cmd = mock_run.call_args[0][0]
    assert cmd.index("-ss") < cmd.index("-i")
    assert cmd[cmd.index("-ss") + 1] == "90.0"

    # input_args must appear contiguously and in order, sandwiched between
    # ffmpeg's leading flags and the trailing -frames:v 1 -vf <scale> <dest>.
    start = cmd.index("-ss")
    assert cmd[start : start + len(input_args)] == input_args
    tail = cmd[start + len(input_args) :]
    assert tail[:2] == ["-frames:v", "1"]
    assert tail[2] == "-vf"
    assert tail[3] == "scale=w=600:h=600:force_original_aspect_ratio=decrease"
    assert tail[4] == str(dest)


def test_normalize_embedded_cover_map_follows_its_i(tmp_path):
    dest = tmp_path / "cover.jpg"
    input_args = ["-i", "/v.mkv", "-map", "0:3"]
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        dest.write_bytes(b"jpegdata")
        _normalize(input_args, str(dest))
    cmd = mock_run.call_args[0][0]
    assert cmd.index("-i") < cmd.index("-map")
    assert cmd[cmd.index("-map") + 1] == "0:3"


def test_normalize_false_on_ffmpeg_failure(tmp_path):
    dest = tmp_path / "cover.jpg"
    result = MagicMock(returncode=1, stdout="", stderr="bad image")
    with patch("subprocess.run", return_value=result):
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


def test_normalize_false_when_output_missing(tmp_path):
    """ffmpeg can exit 0 without writing anything usable."""
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


def test_normalize_false_when_output_empty(tmp_path):
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        dest.write_bytes(b"")
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


def test_normalize_does_not_report_success_from_a_stale_file(tmp_path):
    """A leftover file from an earlier failed tier must not count as success."""
    dest = tmp_path / "cover.jpg"
    dest.write_bytes(b"stale bytes from a previous tier")
    result = MagicMock(returncode=1, stdout="", stderr="bad image")
    with patch("subprocess.run", return_value=result):
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False
    assert not dest.exists()


def test_normalize_false_when_ffmpeg_missing(tmp_path):
    dest = tmp_path / "cover.jpg"
    with patch("subprocess.run", side_effect=FileNotFoundError("no ffmpeg")):
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False


# ---------------------------------------------------------------------------
# resolve_artwork
# ---------------------------------------------------------------------------


def test_resolve_uses_sidecar_first(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    _touch(season / "poster.jpg")
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index") as probe:
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    path, source = result
    assert path == str(work / "cover.jpg")
    assert "poster.jpg" in source
    probe.assert_not_called()
    assert str(season / "poster.jpg") in norm.call_args[0][0]


def test_resolve_falls_through_to_embedded_cover(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=3):
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    assert "embedded" in result[1]
    assert "0:3" in norm.call_args[0][0]


def test_resolve_falls_through_to_frame_grab(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=None):
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    assert "frame" in result[1]
    args = norm.call_args[0][0]
    assert "-ss" in args
    assert "-i" in args
    assert args.index("-ss") < args.index("-i")  # input seek, not output seek
    assert args[args.index("-ss") + 1] == "90.0"  # intro_end 30.0 + 60s offset
    assert "0:v:0" in args


def test_resolve_frame_grab_seek_never_negative(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=True) as norm, \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=None):
        resolve_artwork(str(season), "/v.mkv", -500.0, str(work))
    args = norm.call_args[0][0]
    assert args.index("-ss") < args.index("-i")  # input seek, not output seek
    seek = float(args[args.index("-ss") + 1])
    assert seek >= 0.0


def test_resolve_falls_through_when_sidecar_is_corrupt(tmp_path):
    """A sidecar that fails to normalize must not block the embedded tier."""
    season = tmp_path / "Show" / "Season 01"
    _touch(season / "poster.jpg")
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", side_effect=[False, True]), \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=3):
        result = resolve_artwork(str(season), "/v.mkv", 30.0, str(work))
    assert result is not None
    assert "embedded" in result[1]


def test_resolve_returns_none_when_every_tier_fails(tmp_path):
    season = tmp_path / "Show" / "Season 01"
    season.mkdir(parents=True)
    work = tmp_path / "work"
    work.mkdir()
    with patch("tvshow_noisemaker.artwork._normalize", return_value=False), \
         patch("tvshow_noisemaker.artwork._find_embedded_cover_index", return_value=None):
        assert resolve_artwork(str(season), "/v.mkv", 30.0, str(work)) is None


# ---------------------------------------------------------------------------
# UnicodeDecodeError never escapes (artwork must never fail a conversion)
# ---------------------------------------------------------------------------


def test_embedded_cover_none_on_decode_error():
    """A locale that can't decode ffprobe's stderr must not crash the run."""
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    with patch("subprocess.run", side_effect=exc):
        assert _find_embedded_cover_index("/v.mkv") is None


def test_normalize_false_on_decode_error(tmp_path):
    """A locale that can't decode ffmpeg's stderr must not crash the run."""
    dest = tmp_path / "cover.jpg"
    exc = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    with patch("subprocess.run", side_effect=exc):
        assert _normalize(["-i", "/src/poster.jpg"], str(dest)) is False
