#!/usr/bin/env python3
"""Score pipeline.detect against hand-verified cut points.

Usage (from the repo root):

    .venv/bin/python tools/bench_detection.py [--media test_media] [--truth tools/ground_truth.json] [--season NAME]

Walks every folder under --media that directly contains video files, runs
detection on each, and prints per-episode signed errors (detected − truth),
which boundaries would have clipped dialogue, and how many episodes the truth
marks as "no intro" but detection cut with high confidence.  Report only; exit
code is always 0.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvshow_noisemaker.config import Config  # noqa: E402
from tvshow_noisemaker.pipeline import VIDEO_EXTENSIONS, EpisodePlan, detect  # noqa: E402

BOUNDARIES = ("intro_start", "intro_end", "outro_start")
CLIP_TOLERANCE = 0.25  # seconds of lost dialogue we count as a clip


@dataclass
class Row:
    key: str
    confidence: str
    flags: List[str]
    errors: Dict[str, Optional[float]] = field(default_factory=dict)  # detected − truth
    clipped: List[str] = field(default_factory=list)
    undetected: List[str] = field(default_factory=list)
    missed_no_intro: bool = False


def key_for(path: str, media_root: str) -> str:
    return Path(path).resolve().relative_to(Path(media_root).resolve()).as_posix()


def season_dirs(media_root: str) -> List[str]:
    root = Path(media_root)
    found = {
        str(p.parent) for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    }
    return sorted(found)


def compare_episode(ep: EpisodePlan, truth: dict, tolerance: float = CLIP_TOLERANCE) -> Row:
    row = Row(key=ep.basename, confidence=ep.confidence, flags=list(ep.flags))
    for name in BOUNDARIES:
        if name not in truth:
            continue
        expected = truth[name]
        detected = getattr(ep, name)
        if expected is None or detected is None:
            row.errors[name] = None
            if expected is not None and detected is None:
                row.undetected.append(name)
            continue
        err = detected - expected
        row.errors[name] = err
        # Dialogue is lost when the intro ends later than it really does, the
        # credits are cut earlier than they really start, or a cold open is cut
        # short by an early intro_start.
        loses_dialogue = (
            (name == "intro_end" and err > tolerance)
            or (name == "outro_start" and err < -tolerance)
            or (name == "intro_start" and err < -tolerance)
        )
        if loses_dialogue:
            row.clipped.append(name)
    if "intro_end" in truth and truth["intro_end"] is None:
        row.missed_no_intro = ep.confidence == "high" and ep.intro_end is not None
    return row


def report(rows: List[Row]) -> str:
    lines = [f"{'episode':<60} {'conf':<7} {'i_start':>8} {'i_end':>8} {'outro':>8}  notes"]
    for r in rows:
        cells = []
        for name in BOUNDARIES:
            err = r.errors.get(name)
            cells.append(f"{err:+8.2f}" if err is not None else f"{'-':>8}")
        notes = []
        if r.clipped:
            notes.append("CLIPPED " + ",".join(r.clipped))
        if r.undetected:
            notes.append("UNDETECTED " + ",".join(r.undetected))
        if r.missed_no_intro:
            notes.append("NO-INTRO CUT")
        notes.extend(r.flags)
        lines.append(f"{r.key[:60]:<60} {r.confidence:<7} {' '.join(cells)}  {'; '.join(notes)}")
    lines.append("")
    lines.append(f"episodes compared: {len(rows)}")
    lines.append(f"clipped dialogue: {sum(1 for r in rows if r.clipped)}")
    lines.append(f"undetected boundaries: {sum(1 for r in rows if r.undetected)}")
    lines.append(f"no-intro episodes cut anyway: {sum(1 for r in rows if r.missed_no_intro)}")
    for name in BOUNDARIES:
        errs = [abs(r.errors[name]) for r in rows if r.errors.get(name) is not None]
        if errs:
            lines.append(
                f"max |{name}| error: {max(errs):.2f}s   mean: {sum(errs) / len(errs):.2f}s   n={len(errs)}"
            )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media", default="test_media")
    parser.add_argument("--truth", default="tools/ground_truth.json")
    parser.add_argument("--season", default=None, help="Only seasons whose path contains this text")
    args = parser.parse_args(argv)

    truth = {k: v for k, v in json.loads(Path(args.truth).read_text(encoding="utf-8")).items() if not k.startswith("_")}
    rows: List[Row] = []
    unmatched = 0
    for season in season_dirs(args.media):
        if args.season and args.season not in season:
            continue
        t0 = time.monotonic()
        plan = detect(season, Config())
        print(f"\n== {key_for(season, args.media)}: {len(plan.episodes)} episodes, "
              f"detect {time.monotonic() - t0:.1f}s, stages {plan.stage_timings}")
        for ep in plan.episodes:
            key = key_for(ep.path, args.media)
            if key not in truth:
                unmatched += 1
                continue
            row = compare_episode(ep, truth[key])
            row.key = key
            rows.append(row)
    print()
    print(report(rows))
    print(f"episodes without ground truth: {unmatched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
