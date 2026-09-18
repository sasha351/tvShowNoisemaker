"""Command-line interface for tvshow_noisemaker.

Usage
-----
    tvshow-noisemaker --input-dir /path/to/episodes --output-dir /path/to/mp3s [options]

Detection strategy (in priority order)
---------------------------------------
1. Per-file timestamps supplied via ``--config`` JSON file.
2. Audio fingerprinting across the batch (Chromaprint, chroma STFT fallback).
    Runs in parallel and uses a local on-disk cache for faster repeat runs.
3. Silence detection (ffmpeg ``silencedetect`` filter).
4. User-supplied ``--intro-duration`` / ``--outro-duration`` flags.
5. If unresolved, skip file (avoid full-episode conversion).
"""

from __future__ import annotations

import argparse
import datetime
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

from tvshow_noisemaker.config import VALID_OUTPUT_FORMATS, build_config, load_config
from tvshow_noisemaker.pipeline import (
    Plan,
    detect,
    encode,
    encode_workers,
    fmt_bytes,
    list_video_files,
)


def _build_log_filename(input_dir: str) -> str:
    """Build a log filename from the input folder name and current timestamp."""
    folder_name = Path(input_dir).name or "batch"
    # Sanitise: keep alphanumeric, hyphens, underscores, dots
    folder_name = re.sub(r"[^\w\-.]", "_", folder_name)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{folder_name}_{ts}.log"


def _setup_logging(
    input_dir: str,
    output_dir: str,
    log_file: Optional[str],
    verbose: bool,
) -> tuple:
    """Configure logging with separate terminal and file handlers.

    Terminal gets summary-level messages (INFO+ by default, or
    DEBUG with --verbose). The log file gets everything (DEBUG).

    Returns ``(terminal_logger, app_logger, log_path)``.
    """
    # Determine log file path
    if log_file:
        log_path = log_file
    else:
        log_path = str(Path(output_dir) / _build_log_filename(input_dir))

    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    # Root logger: capture everything so child loggers (detector, etc.) are
    # written to the log file.
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove any pre-existing handlers (e.g. from repeated test runs)
    root.handlers.clear()

    # File handler always captures full DEBUG detail for troubleshooting.
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Terminal handler – summary only
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)

    # App logger receives detailed logs from module loggers via root propagation.
    app_log = logging.getLogger("tvshow_noisemaker")
    app_log.setLevel(logging.DEBUG)

    console_log = logging.getLogger("tvshow_noisemaker.console")
    console_log.setLevel(logging.DEBUG)
    console_log.propagate = False          # don't double-write to file via root
    console_log.addHandler(console)
    console_log.addHandler(fh)             # also write summary lines to log file

    return console_log, app_log, log_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tvshow-noisemaker",
        description=(
            "Trim TV show intros/outros from a folder of video files "
            "and export trimmed .mp3 files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--input-dir", "-i",
        required=True,
        metavar="DIR",
        help="Directory containing video episode files (mp4, mkv, avi, mov, etc.).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        metavar="DIR",
        help="Directory for output .mp3 files (created if it does not exist).",
    )

    # Optional overrides
    parser.add_argument(
        "--intro-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Fallback intro duration in seconds (used when detection fails).",
    )
    parser.add_argument(
        "--outro-duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Fallback outro duration in seconds from the end of the file.",
    )
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=sorted(VALID_OUTPUT_FORMATS),
        default=None,
        help="Output audio format (default: mp3).",
    )
    parser.add_argument(
        "--bitrate", "--mp3-bitrate",
        dest="mp3_bitrate",
        type=int,
        default=None,
        metavar="KBPS",
        help="Encoding bitrate in kbps for mp3/m4a (default: 256).",
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="FILE",
        help=(
            "Path to a JSON config file.  CLI flags take precedence.  "
            "Supports keys: intro_duration, outro_duration, timestamps, "
            "mp3_bitrate, output_format, silence_db, min_silence_len, "
            "fingerprint_search_window."
        ),
    )

    # Log file
    parser.add_argument(
        "--log-file",
        default=None,
        metavar="FILE",
        help=(
            "Override the log file path. By default a log file is written to "
            "the output directory named <input-folder>_<YYYYMMDD_HHMMSS>.log."
        ),
    )

    # Detection strategy
    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument(
        "--fingerprint",
        dest="fingerprint",
        action="store_true",
        default=True,
        help="Use audio fingerprinting for detection (default).",
    )
    fp_group.add_argument(
        "--no-fingerprint",
        dest="fingerprint",
        action="store_false",
        help="Skip audio fingerprinting; use silence detection or user timestamps.",
    )

    # Advanced silence-detection tuning
    parser.add_argument(
        "--silence-db",
        dest="silence_db",
        type=float,
        default=None,
        metavar="DB",
        help="Silence threshold in dBFS for silence detection (default: -40).",
    )
    parser.add_argument(
        "--min-silence-len",
        dest="min_silence_len",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Minimum silence duration in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--fingerprint-search-window",
        dest="fingerprint_search_window",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "How many seconds from the start/end to examine when fingerprinting "
            "(default: 300)."
        ),
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Number of parallel ffmpeg workers for file processing "
            "(default: half of CPU count).  Use 1 for sequential processing."
        ),
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )

    parser.add_argument(
        "--benchmark-fingerprint",
        action="store_true",
        help="Print fingerprint timing/cache statistics in the terminal summary.",
    )

    parser.add_argument(
        "--detect-only",
        dest="detect_only",
        default=None,
        metavar="FILE",
        help="Run detection only and write the per-episode plan as JSON to FILE; no encoding.",
    )
    parser.add_argument(
        "--plan",
        dest="plan",
        default=None,
        metavar="FILE",
        help="Encode from a plan JSON written by --detect-only (skips detection).",
    )

    return parser


def _ensure_ffmpeg_on_path(log: logging.Logger) -> bool:
    """Make sure ffmpeg and ffprobe are available.

    If they are not on PATH, tries to add the static-ffmpeg bundled
    binaries to PATH so that all subprocess calls find them.
    Returns True on success, False if ffmpeg cannot be found.
    """
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return True

    try:
        import static_ffmpeg  # noqa: F811
        static_ffmpeg.add_paths()
        log.info("Using bundled ffmpeg from static-ffmpeg package.")
        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            return True
    except Exception:  # pylint: disable=broad-except
        pass

    log.error(
        "ffmpeg/ffprobe not found. They are installed automatically via "
        "the static-ffmpeg package (pip install static-ffmpeg), or you can "
        "install them manually (e.g. `brew install ffmpeg` on macOS)."
    )
    return False


def run(args: argparse.Namespace) -> int:
    """Main processing logic.  Returns an exit code (0 = success)."""
    run_start = time.monotonic()

    # Create output dir early so we can write the log file there
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con, app_log, log_path = _setup_logging(
        args.input_dir, args.output_dir, args.log_file, getattr(args, "verbose", False),
    )

    if not _ensure_ffmpeg_on_path(con):
        return 1

    try:
        file_data = load_config(args.config)
    except ValueError as exc:
        con.error("Config error: %s", exc)
        return 1
    cfg = build_config(args, file_data)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        con.error("input-dir does not exist or is not a directory: %s", input_dir)
        return 1

    # ── Detect (or load a saved plan) ──────────────────────────────
    plan_file = getattr(args, "plan", None)
    if plan_file:
        try:
            plan = Plan.load(plan_file)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            con.error("Could not load plan %s: %s", plan_file, exc)
            return 1
        con.info("Loaded plan for %d episode(s) from %s", len(plan.episodes), plan_file)
    else:
        video_files = list_video_files(str(input_dir))
        if not video_files:
            con.warning("No video files found in %s", input_dir)
            return 0
        con.info("Found %d video file(s) in %s", len(video_files), input_dir)
        if cfg.use_fingerprinting and len(video_files) >= 2:
            con.info("Running audio fingerprinting across %d episodes…", len(video_files))
        plan = detect(str(input_dir), cfg)
        n_review = sum(1 for ep in plan.episodes if ep.confidence == "review")
        if n_review:
            con.warning(
                "%d episode(s) flagged for review; those without complete boundaries are skipped "
                "(reasons in the log file)", n_review,
            )
        if cfg.use_fingerprinting and len(video_files) >= 2 and not plan.fingerprint_stats:
            con.warning(
                "Fingerprinting failed – fell back to silence detection "
                "(results may be less accurate); see the log file"
            )
        if getattr(args, "benchmark_fingerprint", False) and plan.fingerprint_stats:
            fp_stats = plan.fingerprint_stats
            con.info(
                "Fingerprint benchmark: %.2fs, candidates=%d, jobs=%d, "
                "valid=%d, workers=%d, cache_hits=%d, cache_misses=%d",
                fp_stats.get("elapsed_sec", 0.0),
                int(fp_stats.get("candidate_files", 0)),
                int(fp_stats.get("jobs_submitted", 0)),
                int(fp_stats.get("valid_files", 0)),
                int(fp_stats.get("workers", 0)),
                int(fp_stats.get("cache_hits", 0)),
                int(fp_stats.get("cache_misses", 0)),
            )
            app_log.info("Fingerprint benchmark stats: %s", fp_stats)

    detect_only = getattr(args, "detect_only", None)
    if detect_only:
        plan.save(detect_only)
        n_skip = sum(1 for ep in plan.episodes if ep.confidence == "skip")
        n_review = sum(1 for ep in plan.episodes if ep.confidence == "review")
        con.info(
            "Plan written to %s (%d episodes, %d to encode, %d for review, %d skipped)",
            detect_only, len(plan.episodes), len(plan.to_encode), n_review, n_skip,
        )
        return 0

    # ── Encode ───────────────────────────────────────────────────────
    total_count = len(plan.episodes)
    jobs = plan.to_encode
    batch_start = time.monotonic()
    con.info(
        "Starting conversion  (%d files, %d to encode, %d workers, output → %s)",
        total_count, len(jobs), encode_workers(cfg, len(jobs)), output_dir,
    )

    def _progress(stage: str, done: int, total: int, _message: str) -> None:
        if stage == "encode":
            _print_progress_bar(done, total, batch_start)

    summary = encode(plan, cfg, str(output_dir), progress=_progress)

    # Finish progress line before summary logging
    sys.stdout.write("\n")
    sys.stdout.flush()

    # ── Summary (printed to both terminal and log file) ──────
    batch_elapsed = time.monotonic() - batch_start
    total_elapsed = time.monotonic() - run_start
    rate = (total_count / total_elapsed) if total_elapsed > 0 else 0.0
    timings = plan.stage_timings
    con.info("")
    con.info("═" * 60)
    con.info("  SUMMARY")
    con.info("═" * 60)
    con.info("  Total files found   : %d", total_count)
    con.info("  Converted           : %d", len(summary.converted))
    con.info("  Skipped             : %d", len(summary.skipped))
    con.info("  Failed              : %d", len(summary.failed))
    con.info("─" * 60)
    con.info("  Audio kept          : %s", _fmt_duration(summary.content_secs))
    con.info("  Audio trimmed       : %s", _fmt_duration(summary.trimmed_secs))
    con.info("  Total output size   : %s", fmt_bytes(summary.output_bytes))
    con.info("  Output format       : %s", cfg.output_format)
    if cfg.output_format in ("mp3", "m4a"):
        con.info("  Bitrate             : %d kbps", cfg.mp3_bitrate)
    con.info("  Conversion time     : %s", _fmt_duration(batch_elapsed))
    con.info("  Total runtime       : %s", _fmt_duration(total_elapsed))
    con.info(
        "  Stage timings       : probe=%s, fingerprint=%s, boundaries=%s, encode=%s",
        _fmt_duration(timings.get("probe", 0.0)),
        _fmt_duration(timings.get("fingerprint", 0.0)),
        _fmt_duration(timings.get("boundaries", 0.0)),
        _fmt_duration(summary.encode_sec),
    )
    con.info("  Throughput          : %.2f files/sec", rate)
    con.info("  Output directory    : %s", output_dir)
    con.info("  Log file            : %s", log_path)
    con.info("═" * 60)

    if summary.skipped:
        app_log.warning("Skipped files:")
        for name, reason in summary.skipped:
            app_log.warning("  • %s – %s", name, reason)
    if summary.failed:
        app_log.error("Failed files:")
        for name, reason in summary.failed:
            app_log.error("  • %s – %s", name, reason)
    if summary.skipped or summary.failed:
        con.info("Detailed skipped/failed entries are in the log file.")
    if not summary.failed and not summary.skipped:
        con.info("")
        con.info("All files converted successfully.")

    return 1 if summary.failed else 0


def _fmt_duration(seconds: float) -> str:
    """Format *seconds* as a human-readable string (e.g. '1h 23m 45s')."""
    if seconds < 0:
        return "0s"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if h or m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def _print_progress_bar(done: int, total: int, started_at: float) -> None:
    """Render a single-line progress bar in the terminal."""
    width = 30
    ratio = (done / total) if total > 0 else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.monotonic() - started_at
    rate = (done / elapsed) if elapsed > 0 and done > 0 else 0.0
    remaining = max(total - done, 0)
    eta = (remaining / rate) if rate > 0 else 0.0
    eta_text = _fmt_duration(eta) if done > 0 else "--"
    sys.stdout.write(
        f"\rProgress [{bar}] {done}/{total} "
        f"elapsed={_fmt_duration(elapsed)} eta={eta_text} rate={rate:.2f}/s"
    )
    sys.stdout.flush()


def main() -> None:
    """Entry point installed by ``pyproject.toml``."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
