"""Tests for tvshow_noisemaker.processor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvshow_noisemaker.processor import (
    _aac_encoder,
    _build_artwork_args,
    _codec_args,
    build_output_path,
    process_file,
)


# ---------------------------------------------------------------------------
# build_output_path
# ---------------------------------------------------------------------------


def test_build_output_path_basic():
    result = build_output_path("/input/show/s01e01.mp4", "/output/mp3s")
    assert result == "/output/mp3s/s01e01.mp3"


def test_build_output_path_different_extension():
    result = build_output_path("/data/episode.mp4", "/out")
    assert result == "/out/episode.mp3"


@pytest.mark.parametrize("fmt, ext", [
    ("mp3", ".mp3"), ("m4a", ".m4a"), ("flac", ".flac"), ("wav", ".wav"),
])
def test_build_output_path_formats(fmt, ext):
    result = build_output_path("/data/episode.mp4", "/out", fmt)
    assert result == f"/out/episode{ext}"


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


def test_process_file_raises_if_outro_before_intro():
    with pytest.raises(ValueError, match="must be greater than"):
        process_file(
            input_path="/input/ep01.mp4",
            output_path="/output/ep01.mp3",
            intro_end=100.0,
            outro_start=50.0,
        )


def test_process_file_raises_if_equal_boundaries():
    with pytest.raises(ValueError, match="must be greater than"):
        process_file(
            input_path="/input/ep01.mp4",
            output_path="/output/ep01.mp3",
            intro_end=30.0,
            outro_start=30.0,
        )


def test_process_file_success(tmp_path):
    output_path = str(tmp_path / "ep01.mp3")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=output_path,
            intro_end=30.0,
            outro_start=2550.0,
            mp3_bitrate=192,
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "30.0" in cmd        # -ss intro_end
    assert "2520.0" in cmd      # -t duration (2550 - 30)
    assert "192k" in cmd        # bitrate
    assert output_path in cmd


def test_process_file_writes_metadata(tmp_path):
    output_path = str(tmp_path / "ep01.mp3")
    metadata = {
        "artist": "The Office",
        "album": "Season 2",
        "title": "Episode 03 - Office Olympics",
        "track": "3",
        "genre": "TV Show",
    }
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=output_path,
            intro_end=30.0,
            outro_start=2550.0,
            metadata=metadata,
        )
    cmd = mock_run.call_args[0][0]
    assert "-id3v2_version" in cmd
    assert "artist=The Office" in cmd
    assert "album=Season 2" in cmd
    assert "title=Episode 03 - Office Olympics" in cmd
    assert "track=3" in cmd
    assert "genre=TV Show" in cmd


def test_process_file_creates_output_dir(tmp_path):
    nested_output = str(tmp_path / "nested" / "deep" / "ep01.mp3")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=nested_output,
            intro_end=0.0,
            outro_start=100.0,
        )
    assert Path(nested_output).parent.is_dir()


def test_process_file_raises_on_ffmpeg_failure(tmp_path):
    output_path = str(tmp_path / "ep01.mp3")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="ffmpeg error output"
        )
        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            process_file(
                input_path="/input/ep01.mp4",
                output_path=output_path,
                intro_end=30.0,
                outro_start=2550.0,
            )


def test_process_file_raises_on_timeout(tmp_path):
    import subprocess
    output_path = str(tmp_path / "ep01.mp3")
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 600)):
        with pytest.raises(RuntimeError, match="timed out"):
            process_file(
                input_path="/input/ep01.mp4",
                output_path=output_path,
                intro_end=30.0,
                outro_start=2550.0,
            )


def test_process_file_raises_if_ffmpeg_not_found(tmp_path):
    output_path = str(tmp_path / "ep01.mp3")
    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
        with pytest.raises(RuntimeError, match="ffmpeg executable not found"):
            process_file(
                input_path="/input/ep01.mp4",
                output_path=output_path,
                intro_end=30.0,
                outro_start=2550.0,
            )


@pytest.mark.parametrize("fmt, expected_codec_args, bitrate_expected", [
    ("mp3", ["-acodec", "libmp3lame"], True),
    ("m4a", ["-acodec", "aac"], True),
    ("flac", ["-acodec", "flac"], False),
    ("wav", ["-acodec", "pcm_s16le"], False),
])
def test_process_file_codec_per_format(tmp_path, fmt, expected_codec_args, bitrate_expected):
    output_path = str(tmp_path / f"ep01.{fmt}")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=output_path,
            intro_end=30.0,
            outro_start=2550.0,
            output_format=fmt,
        )
    cmd = mock_run.call_args[0][0]
    idx = cmd.index("-acodec")
    assert cmd[idx:idx + 2] == expected_codec_args
    assert ("-ab" in cmd) == bitrate_expected


def test_process_file_flac_no_id3v2(tmp_path):
    output_path = str(tmp_path / "ep01.flac")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=output_path,
            intro_end=30.0,
            outro_start=2550.0,
            output_format="flac",
            metadata={"artist": "The Office"},
        )
    cmd = mock_run.call_args[0][0]
    assert "-id3v2_version" not in cmd
    assert "artist=The Office" in cmd


def test_process_file_concat_with_cold_open(tmp_path):
    """When intro_start >= 1s and < intro_end, ffmpeg should use filter_complex."""
    output_path = str(tmp_path / "ep01.mp3")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=output_path,
            intro_end=165.0,
            outro_start=2550.0,
            intro_start=120.0,
            mp3_bitrate=192,
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-filter_complex" in cmd
    fc_idx = cmd.index("-filter_complex")
    fc_arg = cmd[fc_idx + 1]
    assert "atrim=start=0:end=120.0" in fc_arg
    assert "atrim=start=165.0:end=2550.0" in fc_arg
    assert "concat=n=2:v=0:a=1" in fc_arg


def test_process_file_simple_when_no_cold_open(tmp_path):
    """When intro_start is 0 (default), use simple seek-based command."""
    output_path = str(tmp_path / "ep01.mp3")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=output_path,
            intro_end=30.0,
            outro_start=2550.0,
            intro_start=0.0,
        )
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert "-filter_complex" not in cmd
    assert "-ss" in cmd
    assert "30.0" in cmd


# ---------------------------------------------------------------------------
# cover artwork
# ---------------------------------------------------------------------------


def _run_process_file(tmp_path, **kwargs):
    """Call process_file with a mocked ffmpeg and return the command list."""
    params = {
        "input_path": "/input/ep01.mp4",
        "output_path": str(tmp_path / "ep01.mp3"),
        "intro_end": 30.0,
        "outro_start": 2550.0,
    }
    params.update(kwargs)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(**params)
    return mock_run.call_args[0][0]


def test_simple_cmd_has_no_vn_flag(tmp_path):
    """-vn silently discards mapped cover art, so it must never be emitted."""
    cmd = _run_process_file(tmp_path)
    assert "-vn" not in cmd


def test_concat_cmd_has_no_vn_flag(tmp_path):
    cmd = _run_process_file(tmp_path, intro_start=165.0, intro_end=200.0)
    assert "-filter_complex" in cmd
    assert "-vn" not in cmd


def test_simple_cmd_embeds_artwork(tmp_path):
    cmd = _run_process_file(tmp_path, artwork_path="/work/cover.jpg")
    assert "/work/cover.jpg" in cmd
    idx = cmd.index("-map", cmd.index("-ac"))
    assert cmd[idx:idx + 2] == ["-map", "1:v"]
    assert "-disposition:v" in cmd
    assert "attached_pic" in cmd
    c_idx = cmd.index("-c:v")
    assert cmd[c_idx:c_idx + 2] == ["-c:v", "copy"]


def test_concat_cmd_embeds_artwork(tmp_path):
    cmd = _run_process_file(
        tmp_path, intro_start=165.0, intro_end=200.0, artwork_path="/work/cover.jpg"
    )
    assert "-filter_complex" in cmd
    assert "/work/cover.jpg" in cmd
    assert "1:v" in cmd
    assert "attached_pic" in cmd


def test_artwork_input_comes_after_source_input(tmp_path):
    """-ss/-t precede input 0 and must not apply to the cover image."""
    cmd = _run_process_file(tmp_path, artwork_path="/work/cover.jpg")
    assert cmd.index("/input/ep01.mp4") < cmd.index("/work/cover.jpg")


def test_no_artwork_args_when_path_is_none(tmp_path):
    cmd = _run_process_file(tmp_path)
    assert "1:v" not in cmd
    assert "-disposition:v" not in cmd


def test_artwork_skipped_for_wav(tmp_path):
    """The wav muxer rejects video streams; mapping an image breaks the encode."""
    cmd = _run_process_file(
        tmp_path,
        output_path=str(tmp_path / "ep01.wav"),
        output_format="wav",
        artwork_path="/work/cover.jpg",
    )
    assert "/work/cover.jpg" not in cmd
    assert "1:v" not in cmd
    assert "-disposition:v" not in cmd


@pytest.mark.parametrize("fmt", ["mp3", "m4a", "flac"])
def test_artwork_embedded_for_non_wav_formats(tmp_path, fmt):
    cmd = _run_process_file(
        tmp_path,
        output_path=str(tmp_path / f"ep01.{fmt}"),
        output_format=fmt,
        artwork_path="/work/cover.jpg",
    )
    assert "/work/cover.jpg" in cmd
    assert "1:v" in cmd


def test_build_artwork_args_returns_empty_pairs_for_wav():
    assert _build_artwork_args("/work/cover.jpg", "wav") == ([], [])


def test_build_artwork_args_returns_empty_pairs_for_none():
    assert _build_artwork_args(None, "mp3") == ([], [])


def test_build_artwork_args_populated_for_mp3():
    inputs, outputs = _build_artwork_args("/work/cover.jpg", "mp3")
    assert inputs == ["-i", "/work/cover.jpg"]
    assert outputs == [
        "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic",
    ]


# ---------------------------------------------------------------------------
# _aac_encoder / _codec_args m4a
# ---------------------------------------------------------------------------


def test_aac_encoder_prefers_audiotoolbox_when_listed():
    listing = " A....D aac                  AAC (Advanced Audio Coding)\n A..... aac_at               aac (AudioToolbox) (codec aac)\n"
    with patch("tvshow_noisemaker.processor.subprocess.run", return_value=MagicMock(returncode=0, stdout=listing)) as run:
        assert _aac_encoder() == "aac_at"
        assert _aac_encoder() == "aac_at"
    assert run.call_count == 1  # probed once per process


def test_aac_encoder_falls_back_to_software_aac():
    with patch("tvshow_noisemaker.processor.subprocess.run", return_value=MagicMock(returncode=0, stdout=" A....D aac  AAC\n")):
        assert _aac_encoder() == "aac"
    with patch("tvshow_noisemaker.processor.subprocess.run", side_effect=OSError("no ffmpeg")):
        _aac_encoder.cache_clear()
        assert _aac_encoder() == "aac"


def test_aac_encoder_falls_back_on_nonzero_returncode():
    listing = " A..... aac_at               aac (AudioToolbox) (codec aac)\n"
    with patch("tvshow_noisemaker.processor.subprocess.run", return_value=MagicMock(returncode=1, stdout=listing)):
        assert _aac_encoder() == "aac"


def test_codec_args_m4a_uses_probed_encoder():
    with patch("tvshow_noisemaker.processor._aac_encoder", return_value="aac_at"):
        assert _codec_args("m4a", 192) == ["-acodec", "aac_at", "-ab", "192k"]
    assert _codec_args("mp3", 192) == ["-acodec", "libmp3lame", "-ab", "192k"]
