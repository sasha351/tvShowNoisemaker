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
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tvshow_noisemaker.config import build_config, load_config
from tvshow_noisemaker.detector import (
    detect_by_fingerprinting,
    get_duration,
    resolve_boundaries,
)
from tvshow_noisemaker.processor import build_output_path, process_file


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
            "Trim TV show intros/outros from a folder of .mp4 files "
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
        help="Directory containing .mp4 episode files.",
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


def run(args: argparse.Namespace) -> int:
    """Main processing logic.  Returns an exit code (0 = success)."""
    log = _setup_logging(args.log_file, getattr(args, "verbose", False))

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

    mp4_files: List[str] = sorted(
        str(p) for p in input_dir.glob("*.mp4")
    )
    if not mp4_files:
        log.warning("No .mp4 files found in %s", input_dir)
        return 0

    log.info("Found %d .mp4 file(s) in %s", len(mp4_files), input_dir)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get durations for all files (needed by fingerprinting and outro detection)
    durations: Dict[str, float] = {}
    for fp in mp4_files:
        dur = get_duration(fp)
        if dur is not None:
            durations[fp] = dur
        else:
            log.warning("Could not determine duration for %s; will attempt to process anyway", fp)

    # Fingerprint pass (batch)
    fingerprint_results: Dict[str, Tuple[float, float]] = {}
    if cfg.use_fingerprinting and len(mp4_files) >= 2:
        log.info("Running audio fingerprinting across %d episodes…", len(mp4_files))
        try:
            fingerprint_results = detect_by_fingerprinting(
                mp4_files,
                durations=durations,
                search_window=cfg.fingerprint_search_window,
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("Fingerprinting failed: %s – falling back to silence detection", exc)

    # Per-file processing
    skipped: List[str] = []
    failed: List[Tuple[str, str]] = []
    processed = 0

    for filepath in mp4_files:
        basename = Path(filepath).name
        log.info("Processing: %s", basename)

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
            log.warning("Skipping %s – %s", basename, reason)
            skipped.append(f"{basename}: {reason}")
            continue

        log.info(
            "  Boundaries – intro_end=%.2fs, outro_start=%.2fs  (duration=%.2fs)",
            intro_end,
            outro_start,
            duration or 0.0,
        )

        if outro_start <= intro_end:
            reason = (
                f"Invalid boundaries: intro_end={intro_end:.2f}s "
                f">= outro_start={outro_start:.2f}s"
            )
            log.warning("Skipping %s – %s", basename, reason)
            skipped.append(f"{basename}: {reason}")
            continue

        output_path = build_output_path(filepath, str(output_dir))

        try:
            process_file(
                input_path=filepath,
                output_path=output_path,
                intro_end=intro_end,
                outro_start=outro_start,
                mp3_bitrate=cfg.mp3_bitrate,
            )
            processed += 1
        except Exception as exc:  # pylint: disable=broad-except
            reason = str(exc)
            log.error("Failed to process %s: %s", basename, reason)
            failed.append((basename, reason))

    # Summary
    log.info(
        "Done. %d processed, %d skipped, %d failed.",
        processed,
        len(skipped),
        len(failed),
    )
    if skipped:
        log.info("Skipped files:")
        for entry in skipped:
            log.info("  • %s", entry)
    if failed:
        log.error("Failed files:")
        for name, reason in failed:
            log.error("  • %s: %s", name, reason)

    return 1 if failed else 0


def main() -> None:
    """Entry point installed by ``pyproject.toml``."""
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
