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
