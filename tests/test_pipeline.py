"""Tests for tvshow_noisemaker.pipeline (all subprocess I/O mocked)."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tvshow_noisemaker.config import Config
from tvshow_noisemaker.detector import MatchInfo, SeasonConsensus
from tvshow_noisemaker.pipeline import (
    EpisodePlan,
    Plan,
    Summary,
    _resolve_episode,
    default_encode_workers,
    detect,
    encode,
    encode_workers,
    list_video_files,
    probe_durations,
)


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


def _consensus(fp, intro_support=2, outro_support=2, intro_len=30.0, outro_len=50.0,
               season_intro=30.0, season_outro=50.0, intro_outliers=(), outro_outliers=()):
    return SeasonConsensus(
        intro_len=season_intro, outro_len=season_outro, intro_episodes=5, outro_episodes=5,
        episodes={fp: MatchInfo(intro_support, outro_support, intro_len, outro_len,
                                list(intro_outliers), list(outro_outliers))},
    )


def test_resolve_episode_high_when_both_boundaries_agree_with_season():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, 2650.0)}, Config(), _consensus(fp))
    assert (ep.intro_start, ep.intro_end, ep.outro_start) == (0.0, 30.0, 2650.0)
    assert ep.confidence == "high" and ep.source == "fingerprint" and ep.flags == []


def test_resolve_episode_no_intro_in_season_keeps_everything_before_credits():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (None, None, 2650.0)}, Config(),
                          _consensus(fp, intro_support=0, intro_len=None, season_intro=None))
    assert (ep.intro_start, ep.intro_end, ep.outro_start) == (None, 0.0, 2650.0)
    assert ep.confidence == "high"
    assert ep.flags == ["no shared intro found in this season"]


def test_resolve_episode_no_credits_in_season_keeps_the_tail():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, None)}, Config(),
                          _consensus(fp, outro_support=0, outro_len=None, season_outro=None))
    assert ep.outro_start == 2700.0 and ep.confidence == "high"
    assert ep.flags == ["no shared credits found in this season"]


def test_resolve_episode_unmatched_intro_is_review_not_filled():
    fp = "/v/ep13.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (None, None, 2650.0)}, Config(),
                          _consensus(fp, intro_support=0, intro_len=None, season_intro=31.2))
    assert ep.intro_start is None and ep.intro_end is None and ep.outro_start == 2650.0
    assert ep.confidence == "review"
    assert ep.flags == ["no intro match (season intro 31 s)"]
    assert ep.encodable is False


def test_resolve_episode_unmatched_credits_is_review():
    fp = "/v/ep05.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, None)}, Config(),
                          _consensus(fp, outro_support=0, outro_len=None, season_outro=39.0))
    assert ep.outro_start is None and ep.confidence == "review"
    assert ep.flags == ["no credits match (season credits 39 s)"]


def test_resolve_episode_single_pair_support_is_review():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, 2650.0)}, Config(),
                          _consensus(fp, intro_support=1, outro_support=1))
    assert ep.confidence == "review"
    assert ep.flags == ["intro matched 1 episode only", "credits matched 1 episode only"]


def test_resolve_episode_length_far_from_season_is_review():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 48.0, 2679.0)}, Config(),
                          _consensus(fp, intro_len=48.0, outro_len=21.0, season_intro=31.0, season_outro=39.0))
    assert ep.confidence == "review"
    assert ep.flags == ["intro length 48 s vs season 31 s", "credits length 21 s vs season 39 s"]


def test_resolve_episode_outliers_are_informational():
    fp = "/v/ep02.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, 2650.0)}, Config(),
                          _consensus(fp, intro_outliers=[45.0], outro_outliers=[12.0, 13.5]))
    assert ep.confidence == "high"
    assert ep.flags == [
        "ignored 1 other shared segment(s) near the start (45 s)",
        "ignored 2 other shared segment(s) near the end (12 s, 14 s)",
    ]


def test_resolve_episode_config_timestamps_win_over_fingerprint():
    fp = "/v/ep01.mkv"
    cfg = Config(timestamps={"ep01.mkv": {"intro_end": 40.0, "outro_start": 2600.0}})
    with patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 40.0, 2600.0)) as rb:
        ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, 2650.0)}, cfg, _consensus(fp))
    rb.assert_called_once()
    assert (ep.intro_end, ep.outro_start, ep.source, ep.confidence) == (40.0, 2600.0, "config", "high")


def test_resolve_episode_without_fingerprint_uses_silence_and_is_review():
    fp = "/v/ep01.mkv"
    with patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 25.0, 2660.0)) as rb:
        ep = _resolve_episode(fp, 2700.0, {}, Config(), SeasonConsensus())
    rb.assert_called_once()
    assert (ep.intro_end, ep.outro_start, ep.source, ep.confidence) == (25.0, 2660.0, "silence", "review")
    assert ep.flags == ["silence detection only (no fingerprint match)"]


def test_resolve_episode_user_durations_are_config_and_high():
    fp = "/v/ep01.mkv"
    with patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2670.0)):
        ep = _resolve_episode(fp, 2700.0, {}, Config(intro_duration=30.0, outro_duration=30.0), SeasonConsensus())
    assert (ep.source, ep.confidence, ep.flags) == ("config", "high", [])


def test_resolve_episode_user_durations_win_over_fingerprint_match():
    fp = "/v/ep01.mkv"
    with patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2670.0)) as rb:
        ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 45.0, 2600.0)}, Config(intro_duration=30.0, outro_duration=30.0),
                              SeasonConsensus(intro_len=45.0, episodes={fp: MatchInfo(intro_support=3)}))
    rb.assert_called_once()
    assert (ep.intro_end, ep.source, ep.confidence) == (30.0, "config", "high")


def test_resolve_episode_nothing_shared_in_season_is_skipped():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (None, None, None)}, Config(),
                          SeasonConsensus(episodes={fp: MatchInfo()}))
    assert ep.confidence == "skip"
    assert ep.flags[0] == "No intro/outro boundaries detected (skipped to avoid full-file conversion)"


def test_resolve_episode_inverted_boundaries_are_skipped():
    fp = "/v/ep01.mkv"
    ep = _resolve_episode(fp, 2700.0, {fp: (0.0, 30.0, 20.0)}, Config(), _consensus(fp))
    assert ep.confidence == "skip" and ep.flags[0].startswith("Invalid boundaries")


def _touch(tmp_path, *names):
    paths = []
    for name in names:
        p = tmp_path / name
        p.touch()
        paths.append(str(p))
    return paths


def test_list_video_files_filters_and_sorts(tmp_path):
    _touch(tmp_path, "b.mkv", "a.mp4", "notes.txt", "._a.mp4", ".DS_Store")
    (tmp_path / "sub").mkdir()
    assert [Path(p).name for p in list_video_files(str(tmp_path))] == ["a.mp4", "b.mkv"]


def test_probe_durations_retries_failed_probe_sequentially(tmp_path):
    files = _touch(tmp_path, "ep01.mp4", "ep02.mp4")

    def _side_effect(fp, timeout=30.0):
        if fp.endswith("ep02.mp4") and timeout == 30.0:
            return None  # fails the parallel pass; retry uses timeout=120.0
        return 2700.0

    with patch("tvshow_noisemaker.pipeline.get_duration", side_effect=_side_effect):
        durations = probe_durations(files)
    assert durations == {files[0]: 2700.0, files[1]: 2700.0}


def test_probe_durations_empty_list_returns_empty_dict():
    assert probe_durations([]) == {}


def test_detect_clamps_nonpositive_workers(tmp_path):
    _touch(tmp_path, "ep01.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False, workers=-1))
    assert len(plan.episodes) == 1


def test_detect_empty_dir_returns_empty_plan(tmp_path):
    plan = detect(str(tmp_path), Config())
    assert plan.episodes == [] and plan.input_dir == str(tmp_path)


def test_detect_probes_fingerprints_and_resolves(tmp_path):
    files = _touch(tmp_path, "ep01.mp4", "ep02.mp4")
    fp_results = {files[0]: (0.0, 30.0, 2650.0)}
    consensus = SeasonConsensus(
        intro_len=30.0, outro_len=50.0, intro_episodes=1, outro_episodes=1,
        episodes={files[0]: MatchInfo(2, 2, 30.0, 50.0)},
    )
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.detect_by_fingerprinting", return_value=fp_results) as mock_fp,
        patch("tvshow_noisemaker.pipeline.get_last_consensus", return_value=consensus),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, 30.0, 2650.0)),
    ):
        plan = detect(str(tmp_path), Config())

    assert [ep.basename for ep in plan.episodes] == ["ep01.mp4", "ep02.mp4"]
    assert plan.episodes[0].source == "fingerprint" and plan.episodes[0].confidence == "high"
    assert plan.episodes[1].source == "silence" and plan.episodes[1].confidence == "review"
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


def test_detect_tolerates_none_boundaries_from_resolver(tmp_path):
    _touch(tmp_path, "ep01.mp4")
    with (
        patch("tvshow_noisemaker.pipeline.get_duration", return_value=2700.0),
        patch("tvshow_noisemaker.pipeline.resolve_boundaries", return_value=(0.0, None, 2650.0)),
    ):
        plan = detect(str(tmp_path), Config(use_fingerprinting=False))
    assert len(plan.episodes) == 1
    assert plan.episodes[0].encodable is False


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


def _plan(tmp_path, n=3, output_format="mp3"):
    names = [f"ep0{i}.mp4" for i in range(1, n + 1)]
    files = _touch(tmp_path, *names)
    return Plan(input_dir=str(tmp_path), episodes=[
        EpisodePlan(path=fp, duration=2700.0, intro_start=0.0, intro_end=30.0, outro_start=2650.0)
        for fp in files
    ])


def test_encode_calls_process_file_per_encodable_episode(tmp_path):
    plan = _plan(tmp_path)
    out = tmp_path / "out"
    with (
        patch("tvshow_noisemaker.pipeline.process_file") as mock_proc,
        patch("tvshow_noisemaker.pipeline.resolve_artwork", return_value=None),
    ):
        summary = encode(plan, Config(workers=2), str(out))

    assert mock_proc.call_count == 3
    kwargs = mock_proc.call_args.kwargs
    assert kwargs["intro_end"] == 30.0 and kwargs["outro_start"] == 2650.0
    assert kwargs["output_path"].startswith(str(out))
    assert kwargs["mp3_bitrate"] == 256 and kwargs["output_format"] == "mp3"
    assert sorted(summary.converted) == ["ep01.mp4", "ep02.mp4", "ep03.mp4"]
    assert summary.content_secs == pytest.approx(3 * 2620.0)
    assert summary.trimmed_secs == pytest.approx(3 * 80.0)
    assert summary.failed == [] and summary.skipped == [] and summary.cancelled is False


def test_encode_tallies_skips_without_calling_ffmpeg(tmp_path):
    plan = _plan(tmp_path)
    plan.episodes[1].confidence = "skip"
    plan.episodes[1].flags = ["No intro/outro boundaries detected"]
    with (
        patch("tvshow_noisemaker.pipeline.process_file") as mock_proc,
        patch("tvshow_noisemaker.pipeline.resolve_artwork", return_value=None),
    ):
        summary = encode(plan, Config(), str(tmp_path / "out"))
    assert mock_proc.call_count == 2
    assert summary.skipped == [("ep02.mp4", "No intro/outro boundaries detected")]


def test_encode_one_failure_does_not_stop_batch(tmp_path):
    plan = _plan(tmp_path)

    def _fail_first(**kwargs):
        if kwargs["input_path"].endswith("ep01.mp4"):
            raise RuntimeError("ffmpeg failed")

    with (
        patch("tvshow_noisemaker.pipeline.process_file", side_effect=_fail_first) as mock_proc,
        patch("tvshow_noisemaker.pipeline.resolve_artwork", return_value=None),
    ):
        summary = encode(plan, Config(workers=1), str(tmp_path / "out"))
    assert mock_proc.call_count == 3
    assert summary.failed == [("ep01.mp4", "ffmpeg failed")]
    assert sorted(summary.converted) == ["ep02.mp4", "ep03.mp4"]


def test_encode_resolves_artwork_once_and_passes_path(tmp_path):
    plan = _plan(tmp_path)
    with (
        patch("tvshow_noisemaker.pipeline.process_file") as mock_proc,
        patch("tvshow_noisemaker.pipeline.resolve_artwork", return_value=("/work/cover.jpg", "embedded cover")) as mock_art,
    ):
        encode(plan, Config(), str(tmp_path / "out"))
    mock_art.assert_called_once()
    input_dir_arg, sample_video, intro_end, workdir = mock_art.call_args.args
    assert input_dir_arg == str(tmp_path) and sample_video.endswith("ep01.mp4")
    assert intro_end == 30.0 and isinstance(workdir, str)
    for c in mock_proc.call_args_list:
        assert c.kwargs["artwork_path"] == "/work/cover.jpg"


def test_encode_skips_artwork_for_wav(tmp_path):
    plan = _plan(tmp_path)
    with (
        patch("tvshow_noisemaker.pipeline.process_file") as mock_proc,
        patch("tvshow_noisemaker.pipeline.resolve_artwork") as mock_art,
    ):
        encode(plan, Config(output_format="wav"), str(tmp_path / "out"))
    mock_art.assert_not_called()
    assert all(c.kwargs["artwork_path"] is None for c in mock_proc.call_args_list)


def test_encode_removes_artwork_temp_dir(tmp_path):
    plan = _plan(tmp_path)
    seen = {}

    def _capture(input_dir, sample, intro_end, workdir):
        seen["workdir"] = workdir
        return None

    with (
        patch("tvshow_noisemaker.pipeline.process_file"),
        patch("tvshow_noisemaker.pipeline.resolve_artwork", side_effect=_capture),
    ):
        encode(plan, Config(), str(tmp_path / "out"))
    assert not Path(seen["workdir"]).exists()


def test_encode_honours_cancel_event(tmp_path):
    plan = _plan(tmp_path)
    cancel = threading.Event()
    cancel.set()
    with (
        patch("tvshow_noisemaker.pipeline.process_file") as mock_proc,
        patch("tvshow_noisemaker.pipeline.resolve_artwork", return_value=None),
    ):
        summary = encode(plan, Config(workers=1), str(tmp_path / "out"), cancel=cancel)
    mock_proc.assert_not_called()
    assert summary.cancelled is True
    assert len(summary.skipped) == 3 and all(reason == "Cancelled" for _, reason in summary.skipped)


def test_encode_reports_progress(tmp_path):
    plan = _plan(tmp_path)
    plan.episodes[2].confidence = "skip"
    events = []
    with (
        patch("tvshow_noisemaker.pipeline.process_file"),
        patch("tvshow_noisemaker.pipeline.resolve_artwork", return_value=None),
    ):
        encode(plan, Config(), str(tmp_path / "out"), progress=lambda *e: events.append(e))
    assert events[0][:3] == ("encode", 1, 3)   # the skip counts as done up front
    assert events[-1][:3] == ("encode", 3, 3)


def test_default_encode_workers_uses_all_cores():
    with patch("tvshow_noisemaker.pipeline.os.cpu_count", return_value=8):
        assert default_encode_workers() == 8


def test_default_encode_workers_unknown_cpu_count():
    with patch("tvshow_noisemaker.pipeline.os.cpu_count", return_value=None):
        assert default_encode_workers() == 2


def test_encode_workers_respects_explicit_and_job_count():
    assert encode_workers(Config(workers=4), n_jobs=10) == 4
    assert encode_workers(Config(workers=4), n_jobs=2) == 2
    assert encode_workers(Config(), n_jobs=0) == 1
