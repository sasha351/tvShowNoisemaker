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
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from tvshow_noisemaker.config import Config
from tvshow_noisemaker.detector import (
    detect_by_fingerprinting,
    get_duration,
    get_last_fingerprint_stats,
    resolve_boundaries,
)

logger = logging.getLogger(__name__)

PLAN_VERSION = 1

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
    """Sorted absolute paths of video files directly inside *input_dir*."""
    return sorted(
        str(p) for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def probe_durations(files: List[str], progress: ProgressFn = _noop_progress) -> Dict[str, float]:
    """Return ``{path: seconds}`` for every file whose duration ffprobe can read.

    Probes in parallel, then retries failures one at a time with a longer
    timeout: cloud-synced or slow disks stall the *first* read of a file.
    """
    durations: Dict[str, float] = {}
    failed: List[str] = []
    workers = min(len(files), max(1, os.cpu_count() or 2))
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
) -> EpisodePlan:
    """Resolve one episode's boundaries and decide whether it can be encoded."""
    ep = EpisodePlan(path=fp, duration=duration)
    try:
        intro_start, intro_end, outro_start = resolve_boundaries(
            fp, cfg=cfg, fingerprint_results=fingerprint_results, duration=duration,
        )
    except Exception as exc:  # pylint: disable=broad-except
        ep.confidence = "skip"
        ep.flags.append(f"Boundary detection error: {exc}")
        return ep
    ep.intro_start, ep.intro_end, ep.outro_start = intro_start, intro_end, outro_start

    ts = cfg.timestamps.get(ep.basename, {})
    has_timestamp_override = ts.get("intro_end") is not None or ts.get("outro_start") is not None
    has_user_duration_override = cfg.intro_duration is not None or cfg.outro_duration is not None
    _, fp_ie, fp_os = fingerprint_results.get(fp, (None, None, None))
    has_fingerprint_boundary = fp_ie is not None or fp_os is not None

    if has_timestamp_override or has_user_duration_override:
        ep.source = "config"
    elif has_fingerprint_boundary:
        ep.source = "fingerprint"
    else:
        ep.source = "silence"

    # No strategy produced boundaries: skip rather than convert the whole file.
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
        ep.confidence = "skip"
        ep.flags.append("No intro/outro boundaries detected (skipped to avoid full-file conversion)")
        return ep

    if outro_start <= intro_end:
        ep.confidence = "skip"
        ep.flags.append(
            f"Invalid boundaries: intro_end={intro_end:.2f}s >= outro_start={outro_start:.2f}s"
        )
    return ep


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
    if ep.confidence == "skip":
        logger.warning("  ✗ Skipped – %s", ep.flags[0] if ep.flags else "not encodable")
        return
    intro_start = ep.intro_start or 0.0
    if intro_start > 1.0 and intro_start < (ep.intro_end or 0.0):
        logger.info(
            "  Cut intro: keep [0–%.1fs] + [%.1fs–%.1fs]  (keeping %.1fs, removing %.1fs)",
            intro_start, ep.intro_end, ep.outro_start, ep.content_duration, ep.trimmed_duration,
        )
    else:
        logger.info(
            "  Trim: %.1fs → %.1fs  (keeping %.1fs, removing %.1fs)",
            ep.intro_end, ep.outro_start, ep.content_duration, ep.trimmed_duration,
        )


def detect(input_dir: str, cfg: Config, progress: ProgressFn = _noop_progress) -> Plan:
    """Probe, fingerprint and resolve boundaries for every video in *input_dir*."""
    plan = Plan(input_dir=str(input_dir))
    files = list_video_files(input_dir)
    if not files:
        return plan

    t0 = time.monotonic()
    durations = probe_durations(files, progress)
    plan.stage_timings["probe"] = time.monotonic() - t0

    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    t0 = time.monotonic()
    if cfg.use_fingerprinting and len(files) >= 2:
        progress("fingerprint", 0, len(files), "Fingerprinting…")
        try:
            fingerprint_results = detect_by_fingerprinting(
                files, durations=durations, search_window=cfg.fingerprint_search_window,
            )
            plan.fingerprint_stats = get_last_fingerprint_stats()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Fingerprinting failed: %s – falling back to silence detection "
                "(results may be less accurate)", exc,
            )
        progress("fingerprint", len(files), len(files), "Fingerprinting done")
    plan.stage_timings["fingerprint"] = time.monotonic() - t0

    t0 = time.monotonic()
    workers = min(cfg.workers or max(1, (os.cpu_count() or 2) // 2), len(files))
    by_path: Dict[str, EpisodePlan] = {}
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(_resolve_episode, fp, durations.get(fp), fingerprint_results, cfg): fp
            for fp in files
        }
        for future in concurrent.futures.as_completed(future_map):
            fp = future_map[future]
            by_path[fp] = future.result()  # _resolve_episode never raises
            done += 1
            progress("boundaries", done, len(files), Path(fp).name)
    plan.episodes = [by_path[fp] for fp in files]
    plan.stage_timings["boundaries"] = time.monotonic() - t0

    for ep in plan.episodes:
        _log_episode(ep, fingerprint_results)
    return plan
