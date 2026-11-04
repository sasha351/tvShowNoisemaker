"""Detect → plan → encode pipeline shared by the CLI and the GUI.

``detect()`` probes durations, fingerprints the batch and resolves per-episode
boundaries into a :class:`Plan`.  ``encode()`` turns a plan into audio files.
Both take a progress callback ``(stage, done, total, message)`` so a caller
can render a bar or a GUI without the library knowing which.
"""

from __future__ import annotations

import json
import concurrent.futures
import logging
import os
import shutil
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from tvshow_noisemaker.artwork import resolve_artwork
from tvshow_noisemaker.config import Config
from tvshow_noisemaker.detector import (
    MatchInfo,
    SeasonConsensus,
    detect_by_fingerprinting,
    get_duration,
    get_last_consensus,
    get_last_fingerprint_stats,
    resolve_boundaries,
)
from tvshow_noisemaker.metadata import infer_track_metadata
from tvshow_noisemaker.processor import build_output_path, process_file

logger = logging.getLogger(__name__)

PLAN_VERSION = 1

_CONSENSUS_LEN_TOLERANCE = 3.0  # s; an episode's raw match must be this close to the season length to be trusted

# Video formats supported by ffmpeg that we accept as input.
VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".mpg", ".mpeg", ".3gp", ".ogv", ".vob",
}

# (stage, done, total, message); stage is "probe" | "fingerprint" | "boundaries" | "encode"
ProgressFn = Callable[[str, int, int, str], None]


def _noop_progress(stage: str, done: int, total: int, message: str) -> None:
    return None


@dataclass
class EpisodePlan:
    """Cut points and detection verdict for one episode."""

    path: str
    duration: Optional[float] = None
    intro_start: Optional[float] = None
    intro_end: Optional[float] = None
    outro_start: Optional[float] = None
    confidence: str = "high"  # "high" | "review" | "skip"
    flags: List[str] = field(default_factory=list)
    source: str = "fingerprint"  # "config" | "fingerprint" | "silence" | "manual"

    @property
    def basename(self) -> str:
        return Path(self.path).name

    @property
    def encodable(self) -> bool:
        return (
            self.confidence != "skip"
            and self.intro_end is not None
            and self.outro_start is not None
            and self.outro_start > self.intro_end
        )

    @property
    def content_duration(self) -> float:
        """Seconds kept: cold open + post-intro content."""
        return (self.intro_start or 0.0) + ((self.outro_start or 0.0) - (self.intro_end or 0.0))

    @property
    def trimmed_duration(self) -> float:
        if self.duration is None:
            return 0.0
        return max(self.duration - self.content_duration, 0.0)


@dataclass
class Plan:
    """Detection output for one input folder; JSON-serialisable."""

    input_dir: str
    episodes: List[EpisodePlan] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    fingerprint_stats: Dict[str, float] = field(default_factory=dict)

    @property
    def to_encode(self) -> List[EpisodePlan]:
        return [ep for ep in self.episodes if ep.encodable]

    def to_json(self) -> str:
        return json.dumps({"version": PLAN_VERSION, **asdict(self)}, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Plan":
        data = json.loads(text)
        if data.get("version") != PLAN_VERSION:
            raise ValueError(f"Unsupported plan version: {data.get('version')!r}")
        return cls(
            input_dir=data["input_dir"],
            episodes=[EpisodePlan(**ep) for ep in data["episodes"]],
            stage_timings=data.get("stage_timings", {}),
            fingerprint_stats=data.get("fingerprint_stats", {}),
        )

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Plan":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


@dataclass
class Summary:
    """Outcome of :func:`encode`."""

    converted: List[str] = field(default_factory=list)
    skipped: List[Tuple[str, str]] = field(default_factory=list)  # (basename, reason)
    failed: List[Tuple[str, str]] = field(default_factory=list)   # (basename, reason)
    content_secs: float = 0.0
    trimmed_secs: float = 0.0
    output_bytes: int = 0
    encode_sec: float = 0.0
    cancelled: bool = False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def list_video_files(input_dir: str) -> List[str]:
    """Sorted absolute paths of video files directly inside *input_dir*.

    Dotfiles are skipped: macOS/iCloud leaves ``._<name>`` AppleDouble
    siblings next to media and they are not playable.
    """
    return sorted(
        str(p) for p in Path(input_dir).iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def probe_durations(files: List[str], progress: ProgressFn = _noop_progress) -> Dict[str, float]:
    """Return ``{path: seconds}`` for every file whose duration ffprobe can read.

    Probes in parallel, then retries failures one at a time with a longer
    timeout: cloud-synced or slow disks stall the *first* read of a file.
    """
    durations: Dict[str, float] = {}
    failed: List[str] = []
    workers = max(1, min(len(files), max(1, os.cpu_count() or 2)))
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(get_duration, fp): fp for fp in files}
        for future in concurrent.futures.as_completed(future_map):
            fp = future_map[future]
            dur = future.result()
            if dur is None:
                failed.append(fp)
            else:
                durations[fp] = dur
            done += 1
            progress("probe", done, len(files), Path(fp).name)

    if failed:
        logger.info(
            "Duration probe failed for %d file(s); retrying sequentially "
            "(slow disk / cloud-synced files can stall the first read)…",
            len(failed),
        )
        for fp in failed:
            dur = get_duration(fp, timeout=120.0)
            if dur is None:
                logger.warning("Could not determine duration for %s; will attempt to process anyway", fp)
            else:
                durations[fp] = dur
    return durations


def _resolve_episode(
    fp: str,
    duration: Optional[float],
    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
    cfg: Config,
    consensus: SeasonConsensus,
) -> EpisodePlan:
    """Resolve one episode's boundaries, confidence and flags."""
    ep = EpisodePlan(path=fp, duration=duration)
    try:
        ts = cfg.timestamps.get(ep.basename, {})
        has_timestamp_override = ts.get("intro_end") is not None or ts.get("outro_start") is not None
        has_user_duration_override = cfg.intro_duration is not None or cfg.outro_duration is not None

        # ponytail: a partial match (e.g. intro matched, credits not) still routes here and
        # ignores the user's duration for the unmatched side; per-boundary precedence would
        # be the upgrade if that turns out to matter in practice.
        has_match = fp in fingerprint_results and any(v is not None for v in fingerprint_results[fp])
        if fp in fingerprint_results and not has_timestamp_override and (has_match or not has_user_duration_override):
            _resolve_from_consensus(ep, fingerprint_results[fp], consensus.episodes.get(fp, MatchInfo()), consensus)
        else:
            ep.intro_start, ep.intro_end, ep.outro_start = resolve_boundaries(
                fp, cfg=cfg, fingerprint_results=fingerprint_results, duration=duration,
            )
            if has_timestamp_override or has_user_duration_override:
                ep.source = "config"
            else:
                ep.source = "silence"
                ep.confidence = "review"
                ep.flags.append("silence detection only (no fingerprint match)")

        # Nothing to cut at either end and no explicit user choice: skip rather than convert the whole file.
        nothing_detected = (
            duration is not None
            and ep.intro_end is not None and abs(ep.intro_end) < 1e-6
            and ep.outro_start is not None and abs(ep.outro_start - duration) < 1e-3
        )
        if nothing_detected and ep.source != "config":
            ep.confidence = "skip"
            # Skip reasons go first: encode() and the log report flags[0].
            ep.flags.insert(0, "No intro/outro boundaries detected (skipped to avoid full-file conversion)")
            return ep

        # A None boundary just leaves `encodable` False; only compare when both are known.
        if ep.intro_end is not None and ep.outro_start is not None and ep.outro_start <= ep.intro_end:
            ep.confidence = "skip"
            ep.flags.insert(
                0, f"Invalid boundaries: intro_end={ep.intro_end:.2f}s >= outro_start={ep.outro_start:.2f}s",
            )
    except Exception as exc:  # pylint: disable=broad-except
        ep.confidence = "skip"
        ep.flags = [f"Boundary detection error: {exc}"]
    return ep


def _resolve_from_consensus(
    ep: EpisodePlan,
    boundaries: Tuple[Optional[float], Optional[float], Optional[float]],
    info: MatchInfo,
    consensus: SeasonConsensus,
) -> None:
    """Apply the decision table: use matched boundaries, flag weak ones, never fill silently."""
    intro_start, intro_end, outro_start = boundaries
    ep.source = "fingerprint"

    if intro_end is None and consensus.intro_len is None:
        ep.intro_start, ep.intro_end = None, 0.0
        ep.flags.append("no shared intro found in this season")
    elif intro_end is None:
        ep.intro_start, ep.intro_end = None, None
        ep.confidence = "review"
        ep.flags.append(f"no intro match (season intro {consensus.intro_len:.0f} s)")
    elif consensus.intro_len is not None:
        ep.intro_start, ep.intro_end = intro_start, intro_end
        if info.intro_support < 2:
            ep.confidence = "review"
            ep.flags.append("intro matched 1 episode only")
        if (
            info.intro_len is not None
            and abs(info.intro_len - consensus.intro_len) > _CONSENSUS_LEN_TOLERANCE
        ):
            ep.confidence = "review"
            ep.flags.append(f"intro length {info.intro_len:.0f} s vs season {consensus.intro_len:.0f} s")
    else:
        # No season consensus on this side (e.g. chroma STFT fallback): nothing to compare against.
        ep.intro_start, ep.intro_end = intro_start, intro_end

    if outro_start is None and consensus.outro_len is None:
        ep.outro_start = ep.duration
        ep.flags.append("no shared credits found in this season")
    elif outro_start is None:
        ep.outro_start = None
        ep.confidence = "review"
        ep.flags.append(f"no credits match (season credits {consensus.outro_len:.0f} s)")
    elif consensus.outro_len is not None:
        ep.outro_start = outro_start
        if info.outro_support < 2:
            ep.confidence = "review"
            ep.flags.append("credits matched 1 episode only")
        if (
            info.outro_len is not None
            and abs(info.outro_len - consensus.outro_len) > _CONSENSUS_LEN_TOLERANCE
        ):
            ep.confidence = "review"
            ep.flags.append(f"credits length {info.outro_len:.0f} s vs season {consensus.outro_len:.0f} s")
    else:
        # No season consensus on this side (e.g. chroma STFT fallback): nothing to compare against.
        ep.outro_start = outro_start

    if info.intro_outliers:
        ep.flags.append(
            f"ignored {len(info.intro_outliers)} other shared segment(s) near the start "
            f"({', '.join(f'{x:.0f} s' for x in info.intro_outliers)})"
        )
    if info.outro_outliers:
        ep.flags.append(
            f"ignored {len(info.outro_outliers)} other shared segment(s) near the end "
            f"({', '.join(f'{x:.0f} s' for x in info.outro_outliers)})"
        )


def _log_episode(
    ep: EpisodePlan,
    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
) -> None:
    """Write the per-episode lines the log file has always carried."""
    logger.info("")
    logger.info("Processing %s", ep.basename)
    fp_is, fp_ie, fp_os = fingerprint_results.get(ep.path, (None, None, None))
    if fp_ie is not None or fp_os is not None:
        logger.info(
            "  Fingerprint boundaries: intro=%.1fs–%.1fs, outro_start=%s",
            fp_is if fp_is is not None else 0.0,
            fp_ie if fp_ie is not None else 0.0,
            f"{fp_os:.1f}s" if fp_os is not None else "N/A",
        )
    flags = ep.flags[1:] if not ep.encodable and ep.flags else ep.flags
    for flag in flags:
        logger.info("  %s %s", "⚠" if ep.confidence != "high" else "·", flag)
    if not ep.encodable:
        logger.warning("  ✗ Skipped – %s", ep.flags[0] if ep.flags else "not encodable")
        return
    intro_start = ep.intro_start or 0.0
    intro_end = ep.intro_end or 0.0
    outro_start = ep.outro_start or 0.0
    if intro_start > 1.0 and intro_start < intro_end:
        logger.info(
            "  Cut intro: keep [0–%.1fs] + [%.1fs–%.1fs]  (keeping %.1fs, removing %.1fs)",
            intro_start, intro_end, outro_start, ep.content_duration, ep.trimmed_duration,
        )
    else:
        logger.info(
            "  Trim: %.1fs → %.1fs  (keeping %.1fs, removing %.1fs)",
            intro_end, outro_start, ep.content_duration, ep.trimmed_duration,
        )


def detect(
    input_dir: str,
    cfg: Config,
    progress: ProgressFn = _noop_progress,
    cancel: Optional[threading.Event] = None,
) -> Plan:
    """Probe, fingerprint and resolve boundaries for every video in *input_dir*."""
    plan = Plan(input_dir=str(input_dir))
    files = list_video_files(input_dir)
    if not files:
        return plan

    t0 = time.monotonic()
    durations = probe_durations(files, progress)
    plan.stage_timings["probe"] = time.monotonic() - t0
    if cancel is not None and cancel.is_set():
        return plan

    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    consensus = SeasonConsensus()
    t0 = time.monotonic()
    if cfg.use_fingerprinting and len(files) >= 2:
        progress("fingerprint", 0, len(files), "Fingerprinting…")
        try:
            fingerprint_results = detect_by_fingerprinting(
                files, durations=durations, search_window=cfg.fingerprint_search_window,
                cut_margin=cfg.cut_margin,
            )
            plan.fingerprint_stats = get_last_fingerprint_stats()
            consensus = get_last_consensus()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Fingerprinting failed: %s – falling back to silence detection "
                "(results may be less accurate)", exc,
            )
        progress("fingerprint", len(files), len(files), "Fingerprinting done")
    plan.stage_timings["fingerprint"] = time.monotonic() - t0
    if cancel is not None and cancel.is_set():
        return plan

    t0 = time.monotonic()
    workers = max(1, min(cfg.workers or max(1, (os.cpu_count() or 2) // 2), len(files)))
    by_path: Dict[str, EpisodePlan] = {}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_resolve_episode, fp, durations.get(fp), fingerprint_results, cfg, consensus): fp
            for fp in files
        }
        for future in concurrent.futures.as_completed(future_map):
            fp = future_map[future]
            try:
                by_path[fp] = future.result()  # _resolve_episode never raises, but guard anyway
            except Exception as exc:  # pylint: disable=broad-except
                by_path[fp] = EpisodePlan(
                    path=fp, duration=durations.get(fp), confidence="skip",
                    flags=[f"Boundary detection error: {exc}"],
                )
            done += 1
            progress("boundaries", done, len(files), Path(fp).name)
    plan.episodes = [by_path[fp] for fp in files]
    plan.stage_timings["boundaries"] = time.monotonic() - t0

    for ep in plan.episodes:
        _log_episode(ep, fingerprint_results)
    return plan


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


class _Cancelled(Exception):
    """Raised inside a worker when the cancel event is set before it starts."""


def default_encode_workers() -> int:
    """libmp3lame is single-threaded per ffmpeg process, so one worker per core."""
    return max(1, os.cpu_count() or 2)


def encode_workers(cfg: Config, n_jobs: int) -> int:
    workers = cfg.workers or default_encode_workers()
    return min(workers, n_jobs) if n_jobs else 1


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def encode(
    plan: Plan,
    cfg: Config,
    output_dir: str,
    progress: ProgressFn = _noop_progress,
    cancel: Optional[threading.Event] = None,
) -> Summary:
    """Encode every encodable episode of *plan* into *output_dir*.

    Episodes marked ``skip`` (or missing boundaries) are tallied as skipped
    without touching ffmpeg.  Setting *cancel* stops new files from starting;
    files already encoding run to completion.
    """
    summary = Summary()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(plan.episodes)
    jobs: List[EpisodePlan] = []
    for ep in plan.episodes:
        if ep.encodable:
            jobs.append(ep)
        else:
            reason = ep.flags[0] if ep.flags else "Not encodable"
            summary.skipped.append((ep.basename, reason))
            progress("encode", 0, total, f"✗ Skipped – {ep.basename}: {reason}")
    lock = threading.Lock()

    def _done_count() -> int:
        return len(summary.converted) + len(summary.skipped) + len(summary.failed)

    artwork_path: Optional[str] = None
    artwork_dir = tempfile.mkdtemp(prefix="tvshow-art-")
    try:
        # One cover per run: every episode in a season shares an album, and
        # probing each file costs seconds.  wav's muxer rejects video streams.
        if jobs and cfg.output_format != "wav":
            resolved = resolve_artwork(plan.input_dir, jobs[0].path, jobs[0].intro_end, artwork_dir)
            if resolved is not None:
                artwork_path, artwork_source = resolved
                logger.info("Artwork: %s", artwork_source)
            else:
                logger.info("Artwork: none found — encoding without cover art")

        progress("encode", _done_count(), total, "Starting")

        def _encode_one(ep: EpisodePlan) -> Tuple[float, str, int]:
            if cancel is not None and cancel.is_set():
                raise _Cancelled()
            output_path = build_output_path(ep.path, str(out_dir), cfg.output_format)
            metadata = infer_track_metadata(ep.path, plan.input_dir)
            started = time.monotonic()
            process_file(
                input_path=ep.path,
                output_path=output_path,
                intro_end=ep.intro_end,
                outro_start=ep.outro_start,
                intro_start=ep.intro_start or 0.0,
                mp3_bitrate=cfg.mp3_bitrate,
                output_format=cfg.output_format,
                metadata=metadata,
                artwork_path=artwork_path,
                stream_copy=cfg.stream_copy,
            )
            return time.monotonic() - started, output_path, _file_size(output_path)

        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=encode_workers(cfg, len(jobs))) as pool:
            future_map = {pool.submit(_encode_one, ep): ep for ep in jobs}
            for future in concurrent.futures.as_completed(future_map):
                ep = future_map[future]
                try:
                    elapsed, output_path, size = future.result()
                    with lock:
                        summary.converted.append(ep.basename)
                        summary.output_bytes += size
                        summary.content_secs += ep.content_duration
                        summary.trimmed_secs += ep.trimmed_duration
                    logger.info(
                        "  ✓ Done in %.1fs → %s (%s)", elapsed, Path(output_path).name, fmt_bytes(size),
                    )
                    message = f"✓ {ep.basename}"
                except _Cancelled:
                    with lock:
                        summary.cancelled = True
                        summary.skipped.append((ep.basename, "Cancelled"))
                    message = f"Cancelled – {ep.basename}"
                except Exception as exc:  # pylint: disable=broad-except
                    if cancel is not None and cancel.is_set():
                        with lock:
                            summary.cancelled = True
                            summary.skipped.append((ep.basename, "Cancelled"))
                        message = f"Cancelled – {ep.basename}"
                    else:
                        logger.error("  ✗ Failed – %s: %s", ep.basename, exc)
                        with lock:
                            summary.failed.append((ep.basename, str(exc)))
                        message = f"✗ Failed – {ep.basename}: {exc}"
                progress("encode", _done_count(), total, message)
        summary.encode_sec = time.monotonic() - t0
    finally:
        shutil.rmtree(artwork_dir, ignore_errors=True)
    return summary


def fmt_bytes(num_bytes: float) -> str:
    """Human-readable size, e.g. ``12.3 MB``."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}" if unit != "B" else f"{int(num_bytes)} B"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
