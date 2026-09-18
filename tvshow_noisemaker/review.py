"""Widget-free pieces of the GUI review flow: time fields, review rows, snippet playback, log relay."""
from __future__ import annotations

import dataclasses
import logging
import os
import queue
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tvshow_noisemaker.pipeline import EpisodePlan, Plan


def parse_time(text: str) -> float:
    """``"1:23.4"`` / ``"0:01:23.4"`` / ``"83.4"`` → seconds. Raises ValueError on anything else."""
    parts = text.strip().split(":")
    if not text.strip() or len(parts) > 3:
        raise ValueError(f"not a time: {text!r}")
    total = 0.0
    for part in parts:
        value = float(part)
        if value < 0:
            raise ValueError(f"negative time: {text!r}")
        total = total * 60 + value
    return total


def format_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return ""
    # ponytail: nudge past exact x.x5 ties — Python's round()/`.1f` round half-to-even
    # (e.g. 10.25 -> "10.2"), which drifts the round-trip past the GUI's 0.05s tolerance.
    minutes, secs = divmod(round(seconds + 1e-9, 1), 60)
    return f"{int(minutes)}:{secs:04.1f}"


def split_plan(plan: Plan, review_all: bool) -> Tuple[List[EpisodePlan], List[EpisodePlan]]:
    """(episodes to encode now, episodes for the review panel)."""
    if review_all:
        return [], list(plan.episodes)
    review = [ep for ep in plan.episodes if ep.confidence == "review"]
    now = [ep for ep in plan.episodes if ep.confidence != "review"]
    return now, review


@dataclass
class ReviewRow:
    episode: EpisodePlan
    intro_start: str
    intro_end: str
    outro_start: str
    skip: bool = False

    @classmethod
    def from_episode(cls, ep: EpisodePlan) -> "ReviewRow":
        return cls(ep, format_time(ep.intro_start), format_time(ep.intro_end), format_time(ep.outro_start))

    def to_episode(self) -> EpisodePlan:
        """A copy of the episode with the edited cuts applied; blank fields stay None."""
        ep = dataclasses.replace(self.episode, flags=list(self.episode.flags))
        edited = False
        for name in ("intro_start", "intro_end", "outro_start"):
            text = getattr(self, name).strip()
            original = getattr(self.episode, name)
            # A field is "edited" only if its text no longer matches the original's own
            # display string — format_time's 0.1s rounding means an untouched value can
            # re-parse a hair off the original float, which must not count as an edit.
            if text == format_time(original):
                value = original
            else:
                edited = True
                value = parse_time(text) if text else None
            setattr(ep, name, value)
        ep.source = "manual" if edited else ep.source
        ep.confidence = "skip" if self.skip else "high"
        return ep


def snippet_cmd(path: str, t: float, out_wav: str, span: float = 6.0) -> List[str]:
    start = max(0.0, t - span / 2)
    return ["ffmpeg", "-y", "-ss", f"{start:.1f}", "-t", f"{span:.1f}", "-i", path, "-vn", "-ac", "2", "-ar", "44100", out_wav]


class SnippetPlayer:
    """Play a short snippet around a cut: ffmpeg → wav in a temp dir → afplay, on a thread."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._dir = tempfile.mkdtemp(prefix="tvshow-snip-")
        self._lock = threading.Lock()
        self._gen = 0  # ponytail: generation counter, bumped by play()/stop(), guards the thread's late Popen

    def play(self, path: str, t: float) -> None:
        self.stop()
        with self._lock:
            self._gen += 1
            gen = self._gen
        threading.Thread(target=self._play_blocking, args=(path, t, gen), daemon=True).start()

    def _play_blocking(self, path: str, t: float, gen: int) -> None:
        wav = os.path.join(self._dir, "snippet.wav")
        subprocess.run(snippet_cmd(path, t, wav), capture_output=True, timeout=60)
        with self._lock:
            if gen != self._gen:
                return  # stop() (or a newer play()) landed while ffmpeg was running
            self._proc = subprocess.Popen(["afplay", wav])  # ponytail: macOS only, like the folder picker
            proc = self._proc
        proc.wait()

    def stop(self) -> None:
        with self._lock:
            self._gen += 1
            proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:  # pylint: disable=broad-except
                pass


class QueueHandler(logging.Handler):
    """Relay log records to the GUI thread as ``("log", line)`` messages."""

    def __init__(self, q: "queue.Queue") -> None:
        super().__init__()
        self._q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._q.put(("log", self.format(record)))
        except Exception:  # pylint: disable=broad-except
            self.handleError(record)
