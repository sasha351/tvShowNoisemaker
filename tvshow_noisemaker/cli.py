"""Command-line interface for tvshow_noisemaker.

Usage
-----
    tvshow-noisemaker --input-dir /path/to/episodes --output-dir /path/to/mp3s [options]

Detection strategy (in priority order)
---------------------------------------
1. Per-file timestamps supplied via ``--config`` JSON file.
2. Audio fingerprinting across all episodes (librosa chroma features).
   Identifies identical/near-identical segments shared by all episodes.
3. Silence detection (ffmpeg ``silencedetect`` filter).
4. User-supplied ``--intro-duration`` / ``--outro-duration`` flags.
5. No trimming (0 / full duration).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tvshow_noisemaker.config import build_config, load_config
from tvshow_noisemaker.detector import (
    detect_by_fingerprinting,
    get_duration,
    resolve_boundaries,
)
from tvshow_noisemaker.processor import build_output_path, process_file

# Video formats supported by ffmpeg that we accept as input.
_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob",
}


def _setup_logging(log_file: Optional[str], verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    return logging.getLogger("tvshow_noisemaker")


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
        help="MP3 encoding bitrate in kbps (default: 192).",
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
        help="Write skip/failure log to this file (in addition to stdout).",
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
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
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
    log = _setup_logging(args.log_file, getattr(args, "verbose", False))

    # Verify ffmpeg/ffprobe are available before doing any work
    if not _ensure_ffmpeg_on_path(log):
        return 1

    # Load and merge config
    try:
        file_data = load_config(args.config)
    except ValueError as exc:
        log.error("Config error: %s", exc)
        return 1

    cfg = build_config(args, file_data)

    # Gather input files
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        log.error("input-dir does not exist or is not a directory: %s", input_dir)
        return 1

    video_files: List[str] = sorted(
        str(p) for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTENSIONS
    )
    if not video_files:
        log.warning("No video files found in %s", input_dir)
        return 0

    log.info("Found %d video file(s) in %s", len(video_files), input_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get durations for all files (needed by fingerprinting and outro detection)
    durations: Dict[str, float] = {}
    for fp in video_files:
        dur = get_duration(fp)
        if dur is not None:
            durations[fp] = dur
        else:
            log.warning("Could not determine duration for %s; will attempt to process anyway", fp)

    # Fingerprint pass (batch)
    fingerprint_results: Dict[str, Tuple[float, float]] = {}
    if cfg.use_fingerprinting and len(video_files) >= 2:
        log.info("Running audio fingerprinting across %d episodes\u2026", len(video_files))
        try:
            fingerprint_results = detect_by_fingerprinting(
                video_files,
                durations=durations,
                search_window=cfg.fingerprint_search_window,
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Fingerprinting failed: %s – falling back to silence detection", exc)

    # Per-file processing
    skipped: List[Tuple[str, str]] = []
    failed: List[Tuple[str, str]] = []
    processed_files: List[str] = []
    total_content_secs = 0.0
    total_trimmed_secs = 0.0
    total_output_bytes = 0
    total_count = len(video_files)
    batch_start = time.monotonic()

    log.info("")
    log.info("═" * 60)
    log.info("Starting conversion  (%d files, output → %s)", total_count, output_dir)
    log.info("═" * 60)

    for idx, filepath in enumerate(video_files, 1):
        basename = Path(filepath).name
        log.info("")
        log.info("[%d/%d] %s", idx, total_count, basename)

        duration = durations.get(filepath)

        try:
            intro_end, outro_start = resolve_boundaries(
                filepath,
                cfg=cfg,
                fingerprint_results=fingerprint_results,
                duration=duration,
            )
        except Exception as exc:  # pylint: disable=broad-except
            reason = f"Boundary detection error: {exc}"
            log.warning("  ✗ Skipped – %s", reason)
            skipped.append((basename, reason))
            continue

        content_duration = outro_start - intro_end
        trimmed = (duration or 0.0) - content_duration

        log.info(
            "  Trim: %.1fs → %.1fs  (keeping %.1fs, removing %.1fs)",
            intro_end,
            outro_start,
            content_duration,
            max(trimmed, 0.0),
        )

        if outro_start <= intro_end:
            reason = (
                f"Invalid boundaries: intro_end={intro_end:.2f}s "
                f">= outro_start={outro_start:.2f}s"
            )
            log.warning("  ✗ Skipped – %s", reason)
            skipped.append((basename, reason))
            continue

        output_path = build_output_path(filepath, str(output_dir))
        file_start = time.monotonic()

        try:
            process_file(
                input_path=filepath,
                output_path=output_path,
                intro_end=intro_end,
                outro_start=outro_start,
                mp3_bitrate=cfg.mp3_bitrate,
            )
        except Exception as exc:  # pylint: disable=broad-except
            reason = str(exc)
            log.error("  ✗ Failed – %s", reason)
            failed.append((basename, reason))
            continue

        file_elapsed = time.monotonic() - file_start
        out_size = _file_size(output_path)
        total_output_bytes += out_size
        total_content_secs += content_duration
        total_trimmed_secs += max(trimmed, 0.0)
        processed_files.append(basename)

        log.info(
            "  ✓ Done in %.1fs → %s (%s)",
            file_elapsed,
            Path(output_path).name,
            _fmt_bytes(out_size),
        )

    # ── Summary ──────────────────────────────────────────────────
    batch_elapsed = time.monotonic() - batch_start
    log.info("")
    log.info("═" * 60)
    log.info("  SUMMARY")
    log.info("═" * 60)
    log.info("  Total files found   : %d", total_count)
    log.info("  Converted           : %d", len(processed_files))
    log.info("  Skipped             : %d", len(skipped))
    log.info("  Failed              : %d", len(failed))
    log.info("─" * 60)
    log.info("  Audio kept          : %s", _fmt_duration(total_content_secs))
    log.info("  Audio trimmed       : %s", _fmt_duration(total_trimmed_secs))
    log.info("  Total output size   : %s", _fmt_bytes(total_output_bytes))
    log.info("  MP3 bitrate         : %d kbps", cfg.mp3_bitrate)
    log.info("  Elapsed time        : %s", _fmt_duration(batch_elapsed))
    log.info("  Output directory    : %s", output_dir)
    log.info("═" * 60)

    if skipped:
        log.info("")
        log.warning("Skipped files:")
        for name, reason in skipped:
            log.warning("  • %s – %s", name, reason)

    if failed:
        log.info("")
        log.error("Failed files:")
        for name, reason in failed:
            log.error("  • %s – %s", name, reason)

    if not failed and not skipped:
        log.info("")
        log.info("All files converted successfully.")

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


def main() -> None:
    """Entry point installed by ``pyproject.toml``."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
