"""Tests for tvshow_noisemaker.cli (integration-style, all I/O mocked)."""

import logging
import threading
import time

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from tvshow_noisemaker.cli import run, _build_parser, _setup_logging, _default_encode_workers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(args_str: str):
    parser = _build_parser()
    return parser.parse_args(args_str.split())


def _make_args(
    input_dir: str,
    output_dir: str,
    intro_duration=None,
    outro_duration=None,
    mp3_bitrate=None,
    output_format=None,
    config=None,
    log_file=None,
    fingerprint=True,
    silence_db=None,
    min_silence_len=None,
    fingerprint_search_window=None,
    benchmark_fingerprint=False,
    verbose=False,
    workers=None,
):
    """Construct a namespace that matches what argparse would produce."""
    class _Args:
        pass
    a = _Args()
    a.input_dir = input_dir
    a.output_dir = output_dir
    a.intro_duration = intro_duration
    a.outro_duration = outro_duration
    a.mp3_bitrate = mp3_bitrate
    a.output_format = output_format
    a.config = config
    a.log_file = log_file
    a.fingerprint = fingerprint
    a.silence_db = silence_db
    a.min_silence_len = min_silence_len
    a.fingerprint_search_window = fingerprint_search_window
    a.benchmark_fingerprint = benchmark_fingerprint
    a.verbose = verbose
    a.workers = workers
    return a


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------


def test_parser_required_args():
    args = _parse("--input-dir /in --output-dir /out")
    assert args.input_dir == "/in"
    assert args.output_dir == "/out"


def test_parser_defaults():
    args = _parse("--input-dir /in --output-dir /out")
    assert args.mp3_bitrate is None  # default is set in build_config
    assert args.fingerprint is True


def test_parser_no_fingerprint_flag():
    args = _parse("--input-dir /in --output-dir /out --no-fingerprint")
    assert args.fingerprint is False


def test_parser_custom_bitrate():
    args = _parse("--input-dir /in --output-dir /out --mp3-bitrate 320")
    assert args.mp3_bitrate == 320


def test_parser_bitrate_alias():
    args = _parse("--input-dir /in --output-dir /out --bitrate 320")
    assert args.mp3_bitrate == 320


def test_parser_format():
    args = _parse("--input-dir /in --output-dir /out --format flac")
    assert args.output_format == "flac"


def test_parser_format_default_none():
    args = _parse("--input-dir /in --output-dir /out")
    assert args.output_format is None


def test_parser_benchmark_fingerprint_flag():
    args = _parse("--input-dir /in --output-dir /out --benchmark-fingerprint")
    assert args.benchmark_fingerprint is True


def test_parser_workers_flag():
    args = _parse("--input-dir /in --output-dir /out --workers 4")
    assert args.workers == 4


def test_parser_workers_default_none():
    args = _parse("--input-dir /in --output-dir /out")
    assert args.workers is None


# ---------------------------------------------------------------------------
# run() – no .mp4 files
# ---------------------------------------------------------------------------


def test_run_warns_no_mp4_files(tmp_path):
    args = _make_args(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"))
    exit_code = run(args)
    assert exit_code == 0


def test_run_input_dir_not_found(tmp_path):
    args = _make_args(
        input_dir=str(tmp_path / "nonexistent"),
        output_dir=str(tmp_path / "out"),
    )
    exit_code = run(args)
    assert exit_code == 1


# ---------------------------------------------------------------------------
# run() – single file, success path
# ---------------------------------------------------------------------------


def test_run_single_file_success(tmp_path):
    show_dir = tmp_path / "The Office" / "Season 2"
    show_dir.mkdir(parents=True)
    ep = show_dir / "The.Office.S02E03.Office.Olympics.mp4"
    ep.touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(show_dir),
        output_dir=str(out_dir),
        fingerprint=False,  # skip fingerprinting (requires ≥2 files)
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    mock_proc.assert_called_once()
    _, kwargs = mock_proc.call_args
    assert kwargs["input_path"] == str(ep)
    assert kwargs["metadata"] == {
        "artist": "The Office",
        "album": "Season 2",
        "title": "Episode 03 - Office Olympics",
        "track": "3",
        "genre": "TV Show",
    }


def test_run_single_file_process_failure(tmp_path):
    ep = tmp_path / "ep01.mp4"
    ep.touch()
    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file", side_effect=RuntimeError("ffmpeg failed")),
    ):
        exit_code = run(args)

    # At least one failure → non-zero exit
    assert exit_code != 0


# ---------------------------------------------------------------------------
# run() – invalid boundaries skipped
# ---------------------------------------------------------------------------


def test_run_skips_invalid_boundaries(tmp_path):
    ep = tmp_path / "ep01.mp4"
    ep.touch()
    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
    )

    # intro_end >= outro_start → should be skipped
    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 3000.0, 100.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    # Skipped files don't cause a non-zero exit
    assert exit_code == 0
    mock_proc.assert_not_called()


def test_run_skips_unresolved_default_boundaries(tmp_path):
    ep = tmp_path / "ep01.mp4"
    ep.touch()
    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 0.0, 2700.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    mock_proc.assert_not_called()


def test_run_allows_explicit_user_no_trim(tmp_path):
    ep = tmp_path / "ep01.mp4"
    ep.touch()
    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
        intro_duration=0.0,
        outro_duration=0.0,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 0.0, 2700.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    mock_proc.assert_called_once()


# ---------------------------------------------------------------------------
# run() – config file integration
# ---------------------------------------------------------------------------


def test_run_with_config_file(tmp_path):
    ep = tmp_path / "ep01.mp4"
    ep.touch()
    out_dir = tmp_path / "out"

    cfg_data = {
        "timestamps": {"ep01.mp4": {"intro_end": 30.0, "outro_start": 2650.0}},
        "mp3_bitrate": 128,
    }
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(cfg_data), encoding="utf-8")

    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        config=str(cfg_file),
        fingerprint=False,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    # process_file should be called with 128kbps from config
    _, kwargs = mock_proc.call_args
    assert kwargs.get("mp3_bitrate") == 128 or mock_proc.call_args[1].get("mp3_bitrate") == 128


# ---------------------------------------------------------------------------
# run() – log file written
# ---------------------------------------------------------------------------


def test_run_creates_log_file(tmp_path):
    ep = tmp_path / "ep01.mp4"
    ep.touch()
    out_dir = tmp_path / "out"
    log_file = tmp_path / "logs" / "run.log"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        log_file=str(log_file),
        fingerprint=False,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file"),
    ):
        run(args)

    assert log_file.exists()


def test_setup_logging_default_console_level_info(tmp_path):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    con, _, _ = _setup_logging(
        input_dir=str(tmp_path),
        output_dir=str(output_dir),
        log_file=None,
        verbose=False,
    )

    stream_handlers = [h for h in con.handlers if hasattr(h, "stream")]
    assert stream_handlers, "Expected at least one stream handler on console logger"
    # Stream handler level should default to INFO unless --verbose is set.
    assert any(h.level == logging.INFO for h in stream_handlers)


def test_setup_logging_file_handler_always_debug(tmp_path):
    """The log file must capture DEBUG even without --verbose."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(
        input_dir=str(tmp_path),
        output_dir=str(output_dir),
        log_file=None,
        verbose=False,
    )
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "Expected a FileHandler on the root logger"
    assert all(h.level == logging.DEBUG for h in file_handlers)


# ---------------------------------------------------------------------------
# run() – parallel processing
# ---------------------------------------------------------------------------


def test_run_parallel_multiple_files(tmp_path):
    """Verify that multiple files are processed and workers > 1 works."""
    for name in ("ep01.mp4", "ep02.mp4", "ep03.mp4"):
        (tmp_path / name).touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
        workers=2,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    assert mock_proc.call_count == 3


def test_run_retries_failed_duration_probe(tmp_path):
    """A file that fails the first (parallel) duration probe but succeeds on
    the sequential retry should still be processed, not silently excluded."""
    for name in ("ep01.mp4", "ep02.mp4"):
        (tmp_path / name).touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
        workers=2,
    )

    def _duration_side_effect(fp, timeout=30.0):
        if fp.endswith("ep02.mp4") and timeout == 30.0:
            return None  # fails first pass; retry pass uses timeout=120.0
        return 2700.0

    with (
        patch("tvshow_noisemaker.cli.get_duration", side_effect=_duration_side_effect),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    assert mock_proc.call_count == 2


def test_run_parallelizes_boundary_resolution(tmp_path):
    """Boundary resolution should run concurrently when workers > 1."""
    for name in ("ep01.mp4", "ep02.mp4", "ep03.mp4"):
        (tmp_path / name).touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
        workers=2,
    )

    lock = threading.Lock()
    active = 0
    max_active = 0

    def _slow_resolve(*_args, **_kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return (0.0, 30.0, 2650.0)

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", side_effect=_slow_resolve),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    assert mock_proc.call_count == 3
    assert max_active > 1


def test_run_workers_one_sequential(tmp_path):
    """Workers=1 should still process all files."""
    for name in ("ep01.mp4", "ep02.mp4"):
        (tmp_path / name).touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
        workers=1,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)

    assert exit_code == 0
    assert mock_proc.call_count == 2


def test_run_parallel_one_failure_does_not_crash_batch(tmp_path):
    """One file failing shouldn't prevent others from processing."""
    for name in ("ep01.mp4", "ep02.mp4"):
        (tmp_path / name).touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
        output_dir=str(out_dir),
        fingerprint=False,
        workers=2,
    )

    call_count = 0

    def _fail_first(**kwargs):
        nonlocal call_count
        call_count += 1
        if "ep01" in kwargs.get("input_path", ""):
            raise RuntimeError("ffmpeg failed for ep01")

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.process_file", side_effect=_fail_first) as mock_proc,
    ):
        exit_code = run(args)

    # At least one failure → non-zero exit
    assert exit_code != 0
    # But both files should have been attempted
    assert mock_proc.call_count == 2


def test_default_encode_workers_uses_all_cores():
    with patch("tvshow_noisemaker.cli.os.cpu_count", return_value=8):
        assert _default_encode_workers() == 8


def test_default_encode_workers_unknown_cpu_count():
    with patch("tvshow_noisemaker.cli.os.cpu_count", return_value=None):
        assert _default_encode_workers() == 2


# ---------------------------------------------------------------------------
# cover artwork wiring
# ---------------------------------------------------------------------------


def _artwork_run(tmp_path, resolved, episode_count=3, output_format=None):
    """Run a conversion with artwork resolution patched. Returns (mocks, exit_code)."""
    show_dir = tmp_path / "Family Guy" / "Season 2"
    show_dir.mkdir(parents=True)
    for n in range(1, episode_count + 1):
        (show_dir / f"Family.Guy.S02E{n:02d}.Episode.mp4").touch()

    args = _make_args(
        input_dir=str(show_dir),
        output_dir=str(tmp_path / "out"),
        output_format=output_format,
        fingerprint=False,
    )

    with (
        patch("tvshow_noisemaker.cli.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.cli.detect_by_fingerprinting", return_value={}),
        patch("tvshow_noisemaker.cli.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
        patch("tvshow_noisemaker.cli.resolve_artwork", return_value=resolved) as mock_art,
        patch("tvshow_noisemaker.cli.process_file") as mock_proc,
    ):
        exit_code = run(args)
    return mock_art, mock_proc, exit_code


def test_run_resolves_artwork_once_for_the_whole_run(tmp_path):
    """Probing every episode costs seconds each; one cover serves the season."""
    mock_art, mock_proc, exit_code = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)")
    )
    assert exit_code == 0
    mock_art.assert_called_once()
    assert mock_proc.call_count == 3


def test_run_passes_artwork_path_to_every_encode(tmp_path):
    _, mock_proc, _ = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)")
    )
    for _, kwargs in mock_proc.call_args_list:
        assert kwargs["artwork_path"] == "/work/cover.jpg"


def test_run_resolves_artwork_from_first_job_path_and_intro_end(tmp_path):
    mock_art, _, _ = _artwork_run(
        tmp_path, ("/work/cover.jpg", "embedded cover (stream 4)")
    )
    input_dir_arg, sample_video, intro_end, workdir = mock_art.call_args[0]
    assert sample_video.endswith(".mp4")
    assert intro_end == 30.0
    assert Path(input_dir_arg).name == "Season 2"
    assert isinstance(workdir, str)


def test_run_skips_artwork_entirely_for_wav(tmp_path):
    """The wav muxer rejects video streams, so resolving one is wasted work."""
    mock_art, mock_proc, exit_code = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)"),
        output_format="wav",
    )
    assert exit_code == 0
    mock_art.assert_not_called()
    for _, kwargs in mock_proc.call_args_list:
        assert kwargs["artwork_path"] is None


def test_run_continues_when_no_artwork_is_found(tmp_path):
    mock_art, mock_proc, exit_code = _artwork_run(tmp_path, None)
    assert exit_code == 0
    mock_art.assert_called_once()
    assert mock_proc.call_count == 3
    for _, kwargs in mock_proc.call_args_list:
        assert kwargs["artwork_path"] is None


def test_run_removes_the_artwork_temp_dir(tmp_path):
    mock_art, _, _ = _artwork_run(
        tmp_path, ("/work/cover.jpg", "sidecar image (poster.jpg)")
    )
    workdir = mock_art.call_args[0][3]
    assert not Path(workdir).exists()
