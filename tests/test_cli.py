"""Tests for tvshow_noisemaker.cli (integration-style, all I/O mocked)."""

import logging

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from tvshow_noisemaker.cli import run, _build_parser, _setup_logging


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
    # Create a fake .mp4 file
    ep = tmp_path / "ep01.mp4"
    ep.touch()

    out_dir = tmp_path / "out"
    args = _make_args(
        input_dir=str(tmp_path),
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
    call_kwargs = mock_proc.call_args
    assert str(ep) in str(call_kwargs)


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
