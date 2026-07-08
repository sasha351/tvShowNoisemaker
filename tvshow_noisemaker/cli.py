"""Command-line interface for tvshow_noisemaker.

Usage
-----
    tvshow-noisemaker --input-dir /path/to/episodes --output-dir /path/to/mp3s [options]

Detection strategy (in priority order)
---------------------------------------
1. Per-file timestamps supplied via ``--config`` JSON file.
2. Audio fingerprinting over the first 5 episodes (librosa chroma features).
    Runs in parallel and uses a local on-disk cache for faster repeat runs.
3. Silence detection (ffmpeg ``silencedetect`` filter).
4. User-supplied ``--intro-duration`` / ``--outro-duration`` flags.
5. If unresolved, skip file (avoid full-episode conversion).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tvshow_noisemaker.config import build_config, load_config
from tvshow_noisemaker.detector import (
    detect_by_fingerprinting,
    get_last_fingerprint_stats,
    get_duration,
    resolve_boundaries,
)
from tvshow_noisemaker.metadata import infer_track_metadata
from tvshow_noisemaker.processor import build_output_path, process_file

# Video formats supported by ffmpeg that we accept as input.
_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob",
}


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
        "--mp3-bitrate",
        type=int,
        default=None,
        metavar="KBPS",
        help="MP3 encoding bitrate in kbps (default: 256).",
    )

    # Config file
    parser.add_argument(
        "--config", "-c",
        default=None,
        metavar="FILE",
        help=(
            "Path to a JSON config file.  CLI flags take precedence.  "
            "Supports keys: intro_duration, outro_duration, timestamps, "
            "mp3_bitrate, silence_db, min_silence_len, fingerprint_search_window."
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
    duration_probe_sec = 0.0
    fingerprint_sec = 0.0
    boundary_sec = 0.0
    encode_sec = 0.0

    # Create output dir early so we can write the log file there
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con, app_log, log_path = _setup_logging(
        args.input_dir,
        args.output_dir,
        args.log_file,
        getattr(args, "verbose", False),
    )

    # Verify ffmpeg/ffprobe are available before doing any work
    if not _ensure_ffmpeg_on_path(con):
        return 1

    # Load and merge config
    try:
        file_data = load_config(args.config)
    except ValueError as exc:
        con.error("Config error: %s", exc)
        return 1

    cfg = build_config(args, file_data)

    # Gather input files
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        con.error("input-dir does not exist or is not a directory: %s", input_dir)
        return 1

    video_files: List[str] = sorted(
        str(p) for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )
    if not video_files:
        con.warning("No video files found in %s", input_dir)
        return 0

    con.info("Found %d video file(s) in %s", len(video_files), input_dir)

    # Get durations for all files (needed by fingerprinting and outro detection)
    durations: Dict[str, float] = {}
    dur_workers = min(len(video_files), max(1, (os.cpu_count() or 2)))
    duration_stage_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=dur_workers) as pool:
        future_to_fp = {pool.submit(get_duration, fp): fp for fp in video_files}
        for future in concurrent.futures.as_completed(future_to_fp):
            fp = future_to_fp[future]
            dur = future.result()
            if dur is not None:
                durations[fp] = dur
            else:
                app_log.warning("Could not determine duration for %s; will attempt to process anyway", fp)
    duration_probe_sec = time.monotonic() - duration_stage_start

    # Fingerprint pass (batch)
    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    if cfg.use_fingerprinting and len(video_files) >= 2:
        con.info("Running audio fingerprinting across %d episodes…", len(video_files))
        fingerprint_stage_start = time.monotonic()
        try:
            fingerprint_results = detect_by_fingerprinting(
                video_files,
                durations=durations,
                search_window=cfg.fingerprint_search_window,
            )
            if getattr(args, "benchmark_fingerprint", False):
                fp_stats = get_last_fingerprint_stats()
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
        except Exception as exc:  # pylint: disable=broad-except
            con.warning(
                "Fingerprinting failed: %s – falling back to silence detection "
                "(results may be less accurate)",
                exc,
            )
        finally:
            fingerprint_sec = time.monotonic() - fingerprint_stage_start

    # Per-file processing
    skipped: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []
    processed_files: List[str] = []
    total_content_secs = 0.0
    total_trimmed_secs = 0.0
    total_output_bytes = 0
    total_count = len(video_files)
    batch_start = time.monotonic()

    # ── Phase 1: resolve boundaries for all files (includes silence detection) ──
    # (filepath, basename, intro_start, intro_end, outro_start, content_dur, trimmed)
    jobs: List[Tuple[str, str, float, float, float, float, float]] = []

    def _resolve_one(fp: str) -> Tuple[str, Tuple[float, float, float]]:
        dur = durations.get(fp)
        bounds = resolve_boundaries(
            fp,
            cfg=cfg,
            fingerprint_results=fingerprint_results,
            duration=dur,
        )
        return fp, bounds

    boundary_results: Dict[str, Tuple[float, float, float]] = {}
    boundary_errors: Dict[str, str] = {}
    boundary_workers = cfg.workers or max(1, (os.cpu_count() or 2) // 2)
    boundary_workers = min(boundary_workers, len(video_files))
    boundary_stage_start = time.monotonic()

    if boundary_workers <= 1:
        for filepath in video_files:
            try:
                _, bounds = _resolve_one(filepath)
                boundary_results[filepath] = bounds
            except Exception as exc:  # pylint: disable=broad-except
                boundary_errors[filepath] = f"Boundary detection error: {exc}"
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=boundary_workers) as pool:
            future_map = {pool.submit(_resolve_one, fp): fp for fp in video_files}
            for future in concurrent.futures.as_completed(future_map):
                filepath = future_map[future]
                try:
                    _, bounds = future.result()
                    boundary_results[filepath] = bounds
                except Exception as exc:  # pylint: disable=broad-except
                    boundary_errors[filepath] = f"Boundary detection error: {exc}"

    boundary_sec = time.monotonic() - boundary_stage_start

    for filepath in video_files:
        basename = Path(filepath).name
        app_log.info("")
        app_log.info("Processing %s", basename)

        duration = durations.get(filepath)

        # Log what fingerprinting found for this file (if anything)
        fp_is, fp_ie, fp_os = fingerprint_results.get(filepath, (None, None, None))
        if fp_ie is not None or fp_os is not None:
            app_log.info(
                "  Fingerprint boundaries: intro=%.1fs–%.1fs, outro_start=%s",
                fp_is if fp_is is not None else 0.0,
                fp_ie if fp_ie is not None else 0.0,
                f"{fp_os:.1f}s" if fp_os is not None else "N/A",
            )

        if filepath in boundary_errors:
            reason = boundary_errors[filepath]
            app_log.warning("  ✗ Skipped – %s", reason)
            skipped.append((basename, reason))
            continue

        intro_start, intro_end, outro_start = boundary_results[filepath]

        # If no strategy could determine meaningful boundaries, skip instead of
        # silently converting the full file with intro/outro still included.
        ts = cfg.timestamps.get(basename, {})
        has_timestamp_override = (
            ts.get("intro_end") is not None or ts.get("outro_start") is not None
        )
        has_user_duration_override = (
            cfg.intro_duration is not None or cfg.outro_duration is not None
        )
        has_fingerprint_boundary = fp_ie is not None or fp_os is not None
        is_unresolved_defaults = (
            duration is not None
            and abs(intro_end) < 1e-6
            and abs(outro_start - duration) < 1e-3
        )

        if (
            is_unresolved_defaults
            and not has_timestamp_override
            and not has_user_duration_override
            and not has_fingerprint_boundary
        ):
            reason = "No intro/outro boundaries detected (skipped to avoid full-file conversion)"
            app_log.warning("  ✗ Skipped – %s", reason)
            skipped.append((basename, reason))
            continue

        # Content kept = cold open + post-intro content.
        content_duration = intro_start + (outro_start - intro_end)
        trimmed = (duration or 0.0) - content_duration

        if intro_start > 1.0 and intro_start < intro_end:
            app_log.info(
                "  Cut intro: keep [0–%.1fs] + [%.1fs–%.1fs]  (keeping %.1fs, removing %.1fs)",
                intro_start, intro_end, outro_start,
                content_duration, max(trimmed, 0.0),
            )
        else:
            app_log.info(
                "  Trim: %.1fs → %.1fs  (keeping %.1fs, removing %.1fs)",
                intro_end, outro_start,
                content_duration, max(trimmed, 0.0),
            )

        if outro_start <= intro_end:
            reason = (
                f"Invalid boundaries: intro_end={intro_end:.2f}s "
                f">= outro_start={outro_start:.2f}s"
            )
            app_log.warning("  ✗ Skipped – %s", reason)
            skipped.append((basename, reason))
            continue

        jobs.append((filepath, basename, intro_start, intro_end, outro_start, content_duration, trimmed))

    # ── Phase 2: process files in parallel ──────────────────────────────────
    max_workers = cfg.workers or _default_encode_workers()
    max_workers = min(max_workers, len(jobs)) if jobs else 1
    progress_lock = threading.Lock()
    done_count = len(skipped)  # skipped files count as "done" for progress

    con.info(
        "Starting conversion  (%d files, %d to encode, %d workers, output → %s)",
        total_count, len(jobs), max_workers, output_dir,
    )
    _print_progress_bar(done_count, total_count, 0, len(skipped), 0, batch_start)

    def _encode_one(filepath: str, basename: str, intro_start: float,
                    intro_end: float, outro_start: float,
                    content_duration: float, trimmed: float) -> tuple:
        output_path = build_output_path(filepath, str(output_dir))
        metadata = infer_track_metadata(filepath, str(input_dir))
        file_start = time.monotonic()
        process_file(
            input_path=filepath,
            output_path=output_path,
            intro_end=intro_end,
            outro_start=outro_start,
            intro_start=intro_start,
            mp3_bitrate=cfg.mp3_bitrate,
            metadata=metadata,
        )
        file_elapsed = time.monotonic() - file_start
        out_size = _file_size(output_path)
        return basename, file_elapsed, output_path, out_size, content_duration, trimmed

    encode_stage_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_encode_one, fp, bn, is_, ie, os_, cd, tr): (fp, bn)
            for fp, bn, is_, ie, os_, cd, tr in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            fp, basename = future_map[future]
            try:
                basename, file_elapsed, output_path, out_size, content_duration, trimmed = future.result()
                with progress_lock:
                    total_output_bytes += out_size
                    total_content_secs += content_duration
                    total_trimmed_secs += max(trimmed, 0.0)
                    processed_files.append(basename)

                app_log.info(
                    "  ✓ Done in %.1fs → %s (%s)",
                    file_elapsed,
                    Path(output_path).name,
                    _fmt_bytes(out_size),
                )

            except Exception as exc:  # pylint: disable=broad-except
                reason = str(exc)
                app_log.error("  ✗ Failed – %s: %s", basename, reason)
                with progress_lock:
                    failed.append((basename, reason))

            with progress_lock:
                done_count = len(processed_files) + len(skipped) + len(failed)
                _print_progress_bar(
                    done_count,
                    total_count,
                    len(processed_files),
                    len(skipped),
                    len(failed),
                    batch_start,
                )
    encode_sec = time.monotonic() - encode_stage_start

    # Finish progress line before summary logging
    sys.stdout.write("\n")
    sys.stdout.flush()

    # ── Summary (printed to both terminal and log file) ──────────
    batch_elapsed = time.monotonic() - batch_start
    total_elapsed = time.monotonic() - run_start
    rate = (total_count / total_elapsed) if total_elapsed > 0 else 0.0
    con.info("")
    con.info("═" * 60)
    con.info("  SUMMARY")
    con.info("═" * 60)
    con.info("  Total files found   : %d", total_count)
    con.info("  Converted           : %d", len(processed_files))
    con.info("  Skipped             : %d", len(skipped))
    con.info("  Failed              : %d", len(failed))
    con.info("─" * 60)
    con.info("  Audio kept          : %s", _fmt_duration(total_content_secs))
    con.info("  Audio trimmed       : %s", _fmt_duration(total_trimmed_secs))
    con.info("  Total output size   : %s", _fmt_bytes(total_output_bytes))
    con.info("  MP3 bitrate         : %d kbps", cfg.mp3_bitrate)
    con.info("  Conversion time     : %s", _fmt_duration(batch_elapsed))
    con.info("  Total runtime       : %s", _fmt_duration(total_elapsed))
    con.info(
        "  Stage timings       : probe=%s, fingerprint=%s, boundaries=%s, encode=%s",
        _fmt_duration(duration_probe_sec),
        _fmt_duration(fingerprint_sec),
        _fmt_duration(boundary_sec),
        _fmt_duration(encode_sec),
    )
    con.info("  Throughput          : %.2f files/sec", rate)
    con.info("  Output directory    : %s", output_dir)
    con.info("  Log file            : %s", log_path)
    con.info("═" * 60)

    if skipped:
        app_log.warning("Skipped files:")
        for name, reason in skipped:
            app_log.warning("  • %s – %s", name, reason)

    if failed:
        app_log.error("Failed files:")
        for name, reason in failed:
            app_log.error("  • %s – %s", name, reason)

    if skipped or failed:
        con.info("Detailed skipped/failed entries are in the log file.")

    if not failed and not skipped:
        con.info("")
        con.info("All files converted successfully.")

    return 1 if failed else 0


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


def _print_progress_bar(
    done: int,
    total: int,
    processed: int,
    skipped: int,
    failed: int,
    started_at: float,
) -> None:
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
    msg = (
        f"\rProgress [{bar}] {done}/{total} "
        f"ok={processed} skip={skipped} fail={failed} "
        f"elapsed={_fmt_duration(elapsed)} "
        f"eta={eta_text} rate={rate:.2f}/s"
    )
    sys.stdout.write(msg)
    sys.stdout.flush()


def _fmt_bytes(num_bytes: int) -> str:
    """Format *num_bytes* as a human-readable size string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{num_bytes} B"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def _file_size(path: str) -> int:
    """Return file size in bytes, or 0 if the file doesn't exist."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _default_encode_workers() -> int:
    """Default worker count for the encode stage.

    libmp3lame is effectively single-threaded per ffmpeg process, so
    audio-only encodes scale close to linearly up to the core count.
    """
    return max(1, os.cpu_count() or 2)


def main() -> None:
    """Entry point installed by ``pyproject.toml``."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
