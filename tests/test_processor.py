"""Tests for tvshow_noisemaker.processor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tvshow_noisemaker.processor import build_output_path, process_file


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
# ffmpeg thread policy
# ---------------------------------------------------------------------------


def _run_and_capture_cmd(tmp_path, **kwargs) -> list:
    """Invoke process_file with mocked ffmpeg and return the command it built."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        process_file(
            input_path="/input/ep01.mp4",
            output_path=str(tmp_path / "ep01.mp3"),
            intro_end=165.0,
            outro_start=2550.0,
            **kwargs,
        )
    return mock_run.call_args[0][0]


@pytest.mark.parametrize("intro_start", [0.0, 120.0], ids=["simple", "concat"])
def test_ffmpeg_threads_omitted_by_default(tmp_path, intro_start):
    """Default (None) leaves ffmpeg's own threading untouched — the A/B baseline."""
    cmd = _run_and_capture_cmd(tmp_path, intro_start=intro_start)
    assert "-threads" not in cmd
    assert "-filter_complex_threads" not in cmd


def test_ffmpeg_threads_simple_path_caps_decoder_and_encoder(tmp_path):
    """-threads is emitted both before -i (decoder) and after the codec args."""
    cmd = _run_and_capture_cmd(tmp_path, intro_start=0.0, ffmpeg_threads=2)

    assert cmd.count("-threads") == 2
    for idx in [i for i, arg in enumerate(cmd) if arg == "-threads"]:
        assert cmd[idx + 1] == "2"

    # One before -i (decoder side), one after (encoder side).
    input_idx = cmd.index("-i")
    thread_positions = [i for i, arg in enumerate(cmd) if arg == "-threads"]
    assert thread_positions[0] < input_idx < thread_positions[1]


def test_ffmpeg_threads_concat_path_also_caps_filtergraph(tmp_path):
    """The concat filtergraph otherwise defaults to one thread per core."""
    cmd = _run_and_capture_cmd(tmp_path, intro_start=120.0, ffmpeg_threads=2)

    assert "-filter_complex" in cmd
    assert "-filter_complex_threads" in cmd
    assert cmd[cmd.index("-filter_complex_threads") + 1] == "2"
    assert cmd.count("-threads") == 2


def test_ffmpeg_threads_zero_is_emitted_not_omitted(tmp_path):
    """0 means ffmpeg auto and must be distinguishable from None."""
    cmd = _run_and_capture_cmd(tmp_path, intro_start=0.0, ffmpeg_threads=0)
    assert "-threads" in cmd
    assert cmd[cmd.index("-threads") + 1] == "0"
