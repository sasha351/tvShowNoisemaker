"""Detect → plan → encode pipeline shared by the CLI and the GUI.

``detect()`` probes durations, fingerprints the batch and resolves per-episode
boundaries into a :class:`Plan`.  ``encode()`` turns a plan into audio files.
Both take a progress callback ``(stage, done, total, message)`` so a caller
can render a bar or a GUI without the library knowing which.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
