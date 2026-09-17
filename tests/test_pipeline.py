"""Tests for tvshow_noisemaker.pipeline (all subprocess I/O mocked)."""
from __future__ import annotations

import pytest

from tvshow_noisemaker.pipeline import EpisodePlan, Plan, Summary


def _ep(**kw) -> EpisodePlan:
    base = dict(path="/v/ep01.mkv", duration=1300.0, intro_start=0.0, intro_end=30.0, outro_start=1270.0)
    base.update(kw)
    return EpisodePlan(**base)


def test_episode_basename():
    assert _ep().basename == "ep01.mkv"


def test_episode_encodable_requires_ordered_boundaries():
    assert _ep().encodable is True
    assert _ep(intro_end=30.0, outro_start=20.0).encodable is False
    assert _ep(intro_end=None).encodable is False
    assert _ep(outro_start=None).encodable is False
    assert _ep(confidence="skip").encodable is False
    assert _ep(confidence="review").encodable is True


def test_episode_content_and_trimmed_duration_include_cold_open():
    ep = _ep(intro_start=40.0, intro_end=70.0, outro_start=1270.0)
    assert ep.content_duration == pytest.approx(40.0 + 1200.0)
    assert ep.trimmed_duration == pytest.approx(1300.0 - 1240.0)
    assert _ep(intro_start=None).content_duration == pytest.approx(1240.0)
    assert _ep(duration=None).trimmed_duration == 0.0


def test_plan_json_round_trip(tmp_path):
    plan = Plan(
        input_dir="/v",
        episodes=[
            _ep(flags=["matched 1 episode only"], source="fingerprint"),
            _ep(path="/v/ep02.mkv", intro_end=None, outro_start=None, confidence="skip", flags=["no boundaries"]),
        ],
        stage_timings={"probe": 1.5},
        fingerprint_stats={"cache_hits": 2},
    )
    path = tmp_path / "plan.json"
    plan.save(str(path))
    assert Plan.load(str(path)) == plan
    assert Plan.from_json(plan.to_json()) == plan


def test_plan_rejects_unknown_version():
    with pytest.raises(ValueError):
        Plan.from_json('{"version": 99, "input_dir": "/v", "episodes": []}')


def test_plan_to_encode_excludes_skips_and_unencodable():
    plan = Plan(input_dir="/v", episodes=[_ep(), _ep(confidence="skip"), _ep(intro_end=None)])
    assert plan.to_encode == [plan.episodes[0]]


def test_summary_defaults():
    s = Summary()
    assert s.converted == [] and s.skipped == [] and s.failed == []
    assert s.cancelled is False


import threading
import time
from unittest.mock import patch

from tvshow_noisemaker.config import Config
from tvshow_noisemaker.pipeline import detect, list_video_files, probe_durations


def _touch(tmp_path, *names):
    paths = []
    for name in names:
        p = tmp_path / name
        p.touch()
        paths.append(str(p))
    return paths


def test_list_video_files_filters_and_sorts(tmp_path):
    _touch(tmp_path, "b.mkv", "a.MP4", "notes.txt")
    (tmp_path / "sub").mkdir()
    assert [p.split("/")[-1] for p in list_video_files(str(tmp_path))] == ["a.MP4", "b.mkv"]


def test_probe_durations_retries_failed_probe_sequentially(tmp_path):
    files = _touch(tmp_path, "ep01.mp4", "ep02.mp4")

    def _side_effect(fp, timeout=30.0):
        if fp.endswith("ep02.mp4") and timeout == 30.0:
            return None  # fails the parallel pass; retry uses timeout=120.0
        return 2700.0

    with patch("tvshow_noisemaker.pipeline.get_duration", side_effect=_side_effect):
        durations = probe_durations(files)
    assert durations == {files[0]: 2700.0, files[1]: 2700.0}


def test_detect_empty_dir_returns_empty_plan(tmp_path):
    plan = detect(str(tmp_path), Config())
    assert plan.episodes == [] and plan.input_dir == str(tmp_path)


def test_detect_probes_fingerprints_and_resolves(tmp_path):
    files = _touch(tmp_path, "ep01.mp4", "ep02.mp4")
    fp_results = {files[0]: (0.0, 30.0, 2650.0)}
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.detect_by_fingerprinting", return_value=fp_results) as mock_fp,
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
    ):
        plan = detect(str(tmp_path), Config())

    assert [ep.basename for ep in plan.episodes] == ["ep01.mp4", "ep02.mp4"]
    assert plan.episodes[0].source == "fingerprint"
    assert plan.episodes[1].source == "silence"
    assert all(ep.confidence == "high" for ep in plan.episodes)
    assert all(ep.duration == 2700.0 for ep in plan.episodes)
    assert set(plan.stage_timings) == {"probe", "fingerprint", "boundaries"}
    mock_fp.assert_called_once()
    assert mock_fp.call_args.kwargs["search_window"] == 300.0


def test_detect_skips_fingerprinting_when_disabled_or_single_file(tmp_path):
    _touch(tmp_path, "ep01.mp4", "ep02.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.detect_by_fingerprinting") as mock_fp,
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
    ):
        detect(str(tmp_path), Config(use_fingerprinting=False))
    mock_fp.assert_not_called()


def test_detect_survives_fingerprint_exception(tmp_path):
    _touch(tmp_path, "ep01.mp4", "ep02.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.detect_by_fingerprinting", side_effect=RuntimeError("fpcalc died")),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
    ):
        plan = detect(str(tmp_path), Config())
    assert len(plan.episodes) == 2 and all(ep.encodable for ep in plan.episodes)


def test_detect_skips_unresolved_defaults(tmp_path):
    _touch(tmp_path, "ep01.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 0.0, 2700.0)),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False))
    ep = plan.episodes[0]
    assert ep.confidence == "skip"
    assert ep.flags[0].startswith("No intro/outro boundaries detected")


def test_detect_keeps_explicit_user_no_trim(tmp_path):
    _touch(tmp_path, "ep01.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 0.0, 2700.0)),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False, intro_duration=0.0))
    assert plan.episodes[0].confidence == "high"
    assert plan.episodes[0].source == "config"


def test_detect_skips_inverted_boundaries(tmp_path):
    _touch(tmp_path, "ep01.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 3000.0, 100.0)),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False))
    assert plan.episodes[0].confidence == "skip"
    assert plan.episodes[0].flags[0].startswith("Invalid boundaries")


def test_detect_turns_resolver_exception_into_skip(tmp_path):
    _touch(tmp_path, "ep01.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", side_effect=RuntimeError("boom")),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False))
    assert plan.episodes[0].confidence == "skip"
    assert plan.episodes[0].flags == ["Boundary detection error: boom"]


def test_detect_resolves_boundaries_in_parallel(tmp_path):
    _touch(tmp_path, "ep01.mp4", "ep02.mp4", "ep03.mp4")
    lock = threading.Lock()
    active = max_active = 0

    def _slow_resolve(*_a, **_k):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return (0.0, 30.0, 2650.0)

    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", side_effect=_slow_resolve),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False, workers=2))
    assert len(plan.episodes) == 3 and max_active > 1


def test_detect_reports_progress_per_stage(tmp_path):
    _touch(tmp_path, "ep01.mp4", "ep02.mp4")
    events = []
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
    ):
        detect(str(tmp_path), Config(use_fingerprinting=False), progress=lambda *e: events.append(e))
    stages = {e[0] for e in events}
    assert stages == {"probe", "boundaries"}
    assert ("probe", 2, 2) in {e[:3] for e in events}
    assert ("boundaries", 2, 2) in {e[:3] for e in events}
