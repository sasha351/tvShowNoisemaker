"""Tests for tools/bench_detection.py (pure comparison logic; no media)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from bench_detection import Row, compare_episode, key_for, report, season_dirs  # noqa: E402

from tvshow_noisemaker.pipeline import EpisodePlan


def _ep(**kw):
    base = dict(path="/m/Show/Season 01/e1.mkv", duration=1300.0, intro_start=0.0, intro_end=30.0, outro_start=1270.0)
    base.update(kw)
    return EpisodePlan(**base)


def test_key_for_is_relative_posix():
    assert key_for("/m/Show/Season 01/e1.mkv", "/m") == "Show/Season 01/e1.mkv"


def test_compare_reports_signed_errors():
    row = compare_episode(_ep(), {"intro_start": 0.0, "intro_end": 29.0, "outro_start": 1272.0})
    assert row.errors == {"intro_start": 0.0, "intro_end": 1.0, "outro_start": -2.0}
    assert row.clipped == ["intro_end", "outro_start"]  # cut later than truth / earlier than truth


def test_compare_within_tolerance_is_not_clipped():
    row = compare_episode(_ep(), {"intro_end": 29.9, "outro_start": 1269.9})
    assert row.clipped == []


def test_compare_cold_open_cut_short_is_clipped():
    row = compare_episode(_ep(intro_start=40.0, intro_end=70.0), {"intro_start": 45.0, "intro_end": 75.0})
    assert row.clipped == ["intro_start"]


def test_compare_skips_unlisted_and_uncomparable_boundaries():
    row = compare_episode(_ep(outro_start=None), {"intro_end": 30.0, "outro_start": 1270.0})
    assert row.errors == {"intro_end": 0.0, "outro_start": None}
    assert row.clipped == []
    assert row.undetected == ["outro_start"]


def test_compare_flags_high_confidence_cut_when_truth_says_no_intro():
    row = compare_episode(_ep(), {"intro_start": None, "intro_end": None})
    assert row.missed_no_intro is True
    assert compare_episode(_ep(confidence="review"), {"intro_end": None}).missed_no_intro is False
    assert compare_episode(_ep(intro_end=None, confidence="skip"), {"intro_end": None}).missed_no_intro is False


def test_season_dirs_finds_folders_that_contain_videos(tmp_path):
    (tmp_path / "A" / "Season 01").mkdir(parents=True)
    (tmp_path / "A" / "Season 01" / "e1.mkv").touch()
    (tmp_path / "Season 02").mkdir()
    (tmp_path / "Season 02" / "e1.mp4").touch()
    (tmp_path / "empty").mkdir()
    assert season_dirs(str(tmp_path)) == [str(tmp_path / "A" / "Season 01"), str(tmp_path / "Season 02")]


def test_report_summarises_counts():
    rows = [
        Row(key="a", confidence="high", flags=[], errors={"intro_end": 0.4}, clipped=["intro_end"], missed_no_intro=False),
        Row(key="b", confidence="high", flags=[], errors={"intro_end": -0.2}, clipped=[], missed_no_intro=False),
        Row(key="c", confidence="review", flags=["x"], errors={}, clipped=[], missed_no_intro=True),
        Row(key="d", confidence="high", flags=[], errors={"intro_end": None}, undetected=["intro_end"]),
    ]
    text = report(rows)
    assert "clipped dialogue: 1" in text
    assert "undetected boundaries: 1" in text
    assert "no-intro episodes cut anyway: 1" in text
    assert "max |intro_end| error: 0.40s" in text
    assert "a " in text and "review" in text
