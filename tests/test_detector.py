"""Tests for tvshow_noisemaker.detector."""

import json
import threading
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from tvshow_noisemaker.detector import (
    MatchInfo,
    SeasonConsensus,
    _coalesce,
    _consensus_segments,
    _detect_by_chromaprint,
    _compare_episodes_chromaprint,
    _create_inverted_index,
    _find_alignment_offset,
    _find_common_segment_bounds,
    _find_contiguous_at_shift,
    _find_contiguous_range,
    _match_template_position,
    _otsu_threshold,
    _pairwise_matches,
    _popcount32,
    _frame_similarity,
    _refine_results_with_silence,
    _smooth,
    _snap_to_silence,
    detect_by_silence,
    get_duration,
    get_last_consensus,
    resolve_boundaries,
)
from tvshow_noisemaker.config import Config


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def test_coalesce_returns_first_non_none():
    assert _coalesce(None, None, 3.0, 4.0) == 3.0


def test_coalesce_all_none():
    assert _coalesce(None, None) is None


def test_smooth_uniform():
    arr = np.ones(10, dtype=np.float32)
    result = _smooth(arr, 3)
    # Smoothing a uniform array should yield the same values
    np.testing.assert_allclose(result[2:-2], 1.0, atol=1e-5)


def test_smooth_window_one():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = _smooth(arr, 1)
    np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# Otsu threshold
# ---------------------------------------------------------------------------


def test_otsu_threshold_bimodal():
    """Otsu should split a bimodal distribution near the valley."""
    rng = np.random.RandomState(42)
    low = rng.normal(0.2, 0.05, 700).astype(np.float32)
    high = rng.normal(0.8, 0.05, 300).astype(np.float32)
    arr = np.concatenate([low, high])
    threshold = _otsu_threshold(arr)
    assert 0.35 < threshold < 0.65, f"Expected threshold between 0.35–0.65, got {threshold}"


def test_otsu_threshold_uniform():
    """A constant array should return that constant."""
    arr = np.full(100, 0.5, dtype=np.float32)
    threshold = _otsu_threshold(arr)
    assert abs(threshold - 0.5) < 0.01


def test_otsu_threshold_adapts_to_ratio():
    """Whether theme is 10% or 50% of frames, Otsu should still split."""
    rng = np.random.RandomState(42)
    # Short theme: 10% high, 90% low
    low = rng.normal(0.15, 0.03, 900).astype(np.float32)
    high = rng.normal(0.85, 0.03, 100).astype(np.float32)
    t1 = _otsu_threshold(np.concatenate([low, high]))

    # Long theme: 50% high, 50% low
    low2 = rng.normal(0.15, 0.03, 500).astype(np.float32)
    high2 = rng.normal(0.85, 0.03, 500).astype(np.float32)
    t2 = _otsu_threshold(np.concatenate([low2, high2]))

    # Both should find thresholds in the valley between the two modes
    assert 0.18 < t1 < 0.82, f"Short theme threshold out of range: {t1}"
    assert 0.18 < t2 < 0.82, f"Long theme threshold out of range: {t2}"


def test_frame_similarity_identical():
    fp = np.random.rand(12, 50).astype(np.float32)
    sim = _frame_similarity(fp, fp)
    np.testing.assert_allclose(sim, 1.0, atol=1e-5)


def test_frame_similarity_different_lengths():
    fp_a = np.random.rand(12, 50).astype(np.float32)
    fp_b = np.random.rand(12, 30).astype(np.float32)
    sim = _frame_similarity(fp_a, fp_b)
    assert len(sim) == 30


# ---------------------------------------------------------------------------
# Alignment and common-segment detection
# ---------------------------------------------------------------------------


def test_find_alignment_offset_identical():
    """Identical fingerprints should align at offset 0."""
    fp = np.random.rand(12, 200).astype(np.float32)
    assert _find_alignment_offset(fp, fp, max_lag_frames=100) == 0


def test_find_alignment_offset_shifted():
    """A fingerprint with a known shift should be detected."""
    # Make a fingerprint with a distinctive region at frames 20-60
    rng = np.random.RandomState(42)
    base = rng.rand(12, 200).astype(np.float32) * 0.1
    theme = rng.rand(12, 40).astype(np.float32)

    ref = base.copy()
    ref[:, 20:60] = theme  # theme at 20-60

    other = base.copy()
    other[:, 50:90] = theme  # theme at 50-90, shifted right by 30

    lag = _find_alignment_offset(ref, other, max_lag_frames=100)
    # other[:, i - lag] matches ref[:, i]; theme in other starts 30 later → lag ≈ -30
    assert abs(lag - (-30)) <= 2, f"Expected lag ≈ -30, got {lag}"


def test_find_common_segment_bounds_no_cold_open():
    """When the theme starts at frame 0 in all episodes, should still work."""
    rng = np.random.RandomState(42)
    theme = rng.rand(12, 100).astype(np.float32) + 0.5

    fps = []
    for _ in range(3):
        fp = rng.rand(12, 300).astype(np.float32) * 0.1
        fp[:, :100] = theme + rng.randn(12, 100).astype(np.float32) * 0.02
        fps.append(fp)

    result = _find_common_segment_bounds(fps, sr=11025, min_duration=1.0, max_duration=60.0)
    assert result is not None
    seg_start, seg_len = result
    hop_sec = 512 / 11025
    # Theme spans frames 0–100 in the reference episode.
    assert seg_start * hop_sec < 3.0, f"Expected segment start near 0s, got {seg_start * hop_sec:.1f}s"
    assert abs((seg_start + seg_len) * hop_sec - 100 * hop_sec) < 3.0, (
        f"Expected segment end near {100 * hop_sec:.1f}s, got {(seg_start + seg_len) * hop_sec:.1f}s"
    )


def test_find_common_segment_bounds_with_cold_open():
    """Theme at different positions (cold open) should be detected."""
    rng = np.random.RandomState(42)
    theme = rng.rand(12, 80).astype(np.float32) + 0.5

    # Episode 1: theme at 50-130
    fp1 = rng.rand(12, 400).astype(np.float32) * 0.1
    fp1[:, 50:130] = theme + rng.randn(12, 80).astype(np.float32) * 0.02

    # Episode 2: theme at 80-160
    fp2 = rng.rand(12, 400).astype(np.float32) * 0.1
    fp2[:, 80:160] = theme + rng.randn(12, 80).astype(np.float32) * 0.02

    # Episode 3: theme at 30-110
    fp3 = rng.rand(12, 400).astype(np.float32) * 0.1
    fp3[:, 30:110] = theme + rng.randn(12, 80).astype(np.float32) * 0.02

    result = _find_common_segment_bounds([fp1, fp2, fp3], sr=11025, min_duration=1.0, max_duration=60.0)
    assert result is not None
    seg_start, seg_len = result
    hop_sec = 512 / 11025
    # In reference (fp1) coordinates the theme spans frames 50–130.
    assert abs(seg_start * hop_sec - 50 * hop_sec) < 3.0, (
        f"Expected segment start near {50 * hop_sec:.1f}s, got {seg_start * hop_sec:.1f}s"
    )
    assert abs((seg_start + seg_len) * hop_sec - 130 * hop_sec) < 3.0, (
        f"Expected segment end near {130 * hop_sec:.1f}s, got {(seg_start + seg_len) * hop_sec:.1f}s"
    )


def test_find_common_segment_bounds_short_theme():
    """A very short theme (~15 frames ≈ 0.7s) should still be detected."""
    rng = np.random.RandomState(42)
    theme = rng.rand(12, 15).astype(np.float32) + 0.5

    fps = []
    for offset in (10, 25, 5):
        fp = rng.rand(12, 200).astype(np.float32) * 0.1
        fp[:, offset : offset + 15] = theme + rng.randn(12, 15).astype(np.float32) * 0.02
        fps.append(fp)

    result = _find_common_segment_bounds(fps, sr=11025, min_duration=0.5, max_duration=60.0)
    assert result is not None
    seg_start, seg_len = result
    hop_sec = 512 / 11025
    assert seg_len * hop_sec >= 0.5, f"Detected segment too short: {seg_len * hop_sec:.1f}s"


def test_find_common_segment_bounds_long_theme():
    """A long theme (~200 frames ≈ 9.3s) should be detected in full."""
    rng = np.random.RandomState(42)
    theme = rng.rand(12, 200).astype(np.float32) + 0.5

    fps = []
    for offset in (20, 50, 10):
        fp = rng.rand(12, 500).astype(np.float32) * 0.1
        fp[:, offset : offset + 200] = theme + rng.randn(12, 200).astype(np.float32) * 0.02
        fps.append(fp)

    result = _find_common_segment_bounds(fps, sr=11025, min_duration=1.0, max_duration=60.0)
    assert result is not None
    seg_start, seg_len = result
    hop_sec = 512 / 11025
    # Should capture most of the 200-frame theme (~9.3s)
    assert seg_len * hop_sec > 6.0, f"Long theme not fully captured: {seg_len * hop_sec:.1f}s"


# ---------------------------------------------------------------------------
# _match_template_position
# ---------------------------------------------------------------------------


def test_match_template_position_finds_theme():
    """Template should be found at the correct position in the episode."""
    rng = np.random.RandomState(42)
    theme = rng.rand(12, 80).astype(np.float32) + 0.5

    # Episode: noise with theme at frames 120-200
    episode_fp = rng.rand(12, 400).astype(np.float32) * 0.1
    episode_fp[:, 120:200] = theme + rng.randn(12, 80).astype(np.float32) * 0.02

    result = _match_template_position(theme, episode_fp, sr=11025)
    assert result is not None
    start, end = result
    hop_sec = 512 / 11025
    assert abs(start - 120 * hop_sec) < 2.0
    assert abs(end - 200 * hop_sec) < 2.0


def test_match_template_position_returns_none_for_short_episode():
    """Template longer than episode should return None."""
    template = np.ones((12, 100), dtype=np.float32)
    episode = np.ones((12, 50), dtype=np.float32)
    assert _match_template_position(template, episode, sr=11025) is None


# ---------------------------------------------------------------------------
# get_duration
# ---------------------------------------------------------------------------


def _ffprobe_stdout(duration: float) -> str:
    return json.dumps({"format": {"duration": str(duration)}})


def test_get_duration_success():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_ffprobe_stdout(1234.5),
            stderr="",
        )
        result = get_duration("/fake/ep01.mp4")
    assert result == pytest.approx(1234.5)


def test_get_duration_ffprobe_fails():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="{}", stderr="error")
        result = get_duration("/fake/ep01.mp4")
    assert result is None


def test_get_duration_exception():
    with patch("subprocess.run", side_effect=Exception("boom")):
        result = get_duration("/fake/ep01.mp4")
    assert result is None


def test_get_duration_passes_custom_timeout():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_ffprobe_stdout(1234.5),
            stderr="",
        )
        get_duration("f.mkv", timeout=120.0)
    _, kwargs = mock_run.call_args
    assert kwargs["timeout"] == 120.0


# ---------------------------------------------------------------------------
# detect_by_silence
# ---------------------------------------------------------------------------

_SILENCE_STDERR = """\
[silencedetect @ 0x...] silence_start: 0
[silencedetect @ 0x...] silence_end: 28.5 | silence_duration: 28.5
[silencedetect @ 0x...] silence_start: 2550.0
[silencedetect @ 0x...] silence_end: 2600.0 | silence_duration: 50.0
"""


def _fake_silencedetect_run(head_stderr: str, tail_stderr: str):
    """Return a subprocess.run side_effect that distinguishes head vs tail pass."""
    def _side_effect(cmd, **kwargs):
        stderr = tail_stderr if "-ss" in cmd else head_stderr
        return MagicMock(returncode=0, stdout="", stderr=stderr)
    return _side_effect


def test_detect_by_silence_parses_output():
    # duration=2700 → tail pass seeks to 2100; silence at absolute 2550 is
    # reported by ffmpeg as relative 450.
    tail = (
        "[silencedetect] silence_start: 450.0\n"
        "[silencedetect] silence_end: 500.0 | silence_duration: 50.0\n"
    )
    with patch(
        "subprocess.run",
        side_effect=_fake_silencedetect_run(_SILENCE_STDERR, tail),
    ) as mock_run:
        intro_end, outro_start = detect_by_silence(
            "/fake/ep01.mp4", duration=2700.0
        )
    assert intro_end == pytest.approx(28.5)
    assert outro_start == pytest.approx(2550.0)
    assert mock_run.call_count == 2  # two windowed passes, not one full scan
    head_cmd, tail_cmd = (c.args[0] for c in mock_run.call_args_list)
    assert "-t" in head_cmd and "-ss" not in head_cmd
    assert "-ss" in tail_cmd


def test_detect_by_silence_no_silence():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="", stderr="no silencedetect lines here"
        )
        intro_end, outro_start = detect_by_silence("/fake/ep01.mp4")
    assert intro_end is None
    assert outro_start is None


def test_detect_by_silence_exception():
    with patch("subprocess.run", side_effect=Exception("subprocess crashed")):
        intro_end, outro_start = detect_by_silence("/fake/ep01.mp4")
    assert intro_end is None
    assert outro_start is None


def test_detect_by_silence_outro_in_first_half_ignored():
    """A tail-pass silence that is still in the first 50% of the file is rejected."""
    # duration=1000 → tail seek = 400; relative 50 → absolute 450 < 500 (50%).
    tail = (
        "[silencedetect] silence_start: 50.0\n"
        "[silencedetect] silence_end: 51.0 | silence_duration: 1.0\n"
    )
    with patch(
        "subprocess.run",
        side_effect=_fake_silencedetect_run("no silence", tail),
    ):
        _, outro_start = detect_by_silence("/fake/ep01.mp4", duration=1000.0)
    assert outro_start is None


def test_detect_by_silence_caches_results(tmp_path):
    """Second call with same file should use cache, not ffmpeg."""
    fake_file = tmp_path / "ep01.mp4"
    fake_file.write_bytes(b"\x00" * 100)

    tail = (
        "[silencedetect] silence_start: 450.0\n"
        "[silencedetect] silence_end: 500.0 | silence_duration: 50.0\n"
    )
    with patch(
        "subprocess.run",
        side_effect=_fake_silencedetect_run(_SILENCE_STDERR, tail),
    ) as mock_run:
        r1 = detect_by_silence(str(fake_file), duration=2700.0)

    assert mock_run.call_count == 2  # head + tail pass
    assert r1 == (pytest.approx(28.5), pytest.approx(2550.0))

    with patch("subprocess.run") as mock_run2:
        r2 = detect_by_silence(str(fake_file), duration=2700.0)

    mock_run2.assert_not_called()
    assert r2 == r1


def test_detect_by_silence_skips_video_decode():
    """Every silencedetect ffmpeg invocation must pass -vn (audio-only decode)."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="", stderr=_SILENCE_STDERR,
        )
        detect_by_silence("/fake/ep01.mp4", duration=2700.0)
    assert mock_run.call_args_list, "expected at least one ffmpeg call"
    for c in mock_run.call_args_list:
        assert "-vn" in c.args[0], f"missing -vn in: {c.args[0]}"


def test_detect_by_silence_only_outro_skips_head_pass():
    tail = (
        "[silencedetect] silence_start: 450.0\n"
        "[silencedetect] silence_end: 500.0 | silence_duration: 50.0\n"
    )
    with patch(
        "subprocess.run",
        side_effect=_fake_silencedetect_run("unused", tail),
    ) as mock_run:
        intro_end, outro_start = detect_by_silence(
            "/fake/ep01.mp4", duration=2700.0, need_intro=False,
        )
    assert intro_end is None
    assert outro_start == pytest.approx(2550.0)
    assert mock_run.call_count == 1
    assert "-ss" in mock_run.call_args.args[0]


def test_detect_by_silence_nothing_needed_no_subprocess():
    with patch("subprocess.run") as mock_run:
        result = detect_by_silence(
            "/fake/ep01.mp4", duration=2700.0,
            need_intro=False, need_outro=False,
        )
    mock_run.assert_not_called()
    assert result == (None, None)


def test_resolve_boundaries_partial_fingerprint_limits_silence():
    """Fingerprint found the intro → silence detection only asked for the outro."""
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (0.0, 30.0, None)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (None, 2580.0)
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    kwargs = mock_sil.call_args.kwargs
    assert kwargs["need_intro"] is False
    assert kwargs["need_outro"] is True
    assert intro_end == pytest.approx(30.0)
    assert outro_start == pytest.approx(2580.0)


# ---------------------------------------------------------------------------
# resolve_boundaries
# ---------------------------------------------------------------------------


def _make_cfg(**kwargs) -> Config:
    defaults = dict(
        intro_duration=None,
        outro_duration=None,
        timestamps={},
        mp3_bitrate=192,
        silence_db=-40.0,
        min_silence_len=0.5,
        fingerprint_search_window=300.0,
        use_fingerprinting=True,
        log_file=None,
        workers=None,
    )
    defaults.update(kwargs)
    return Config(**defaults)


def test_resolve_boundaries_uses_config_timestamps():
    cfg = _make_cfg(timestamps={"ep01.mp4": {"intro_end": 30.0, "outro_start": 2600.0}})
    intro_start, intro_end, outro_start = resolve_boundaries(
        "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
    )
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(30.0)
    assert outro_start == pytest.approx(2600.0)


def test_resolve_boundaries_prefers_timestamps_over_fingerprinting():
    cfg = _make_cfg(timestamps={"ep01.mp4": {"intro_end": 10.0, "outro_start": 2000.0}})
    fp_results = {"/show/ep01.mp4": (5.0, 99.0, 500.0)}  # should be ignored
    intro_start, intro_end, outro_start = resolve_boundaries(
        "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
    )
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(10.0)
    assert outro_start == pytest.approx(2000.0)


def test_resolve_boundaries_uses_fingerprinting():
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (5.0, 30.0, 2550.0)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    # Silence detection should NOT be called when fingerprinting covers both boundaries
    mock_sil.assert_not_called()
    assert intro_start == pytest.approx(5.0)
    assert intro_end == pytest.approx(30.0)
    assert outro_start == pytest.approx(2550.0)


def test_resolve_boundaries_falls_back_to_silence():
    cfg = _make_cfg()
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (28.0, 2580.0)
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
        )
    mock_sil.assert_called_once()
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(28.0)
    assert outro_start == pytest.approx(2580.0)


def test_resolve_boundaries_falls_back_to_user_duration():
    cfg = _make_cfg(intro_duration=60.0, outro_duration=30.0)
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (None, None)
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
        )
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(60.0)
    assert outro_start == pytest.approx(2700.0 - 30.0)


def test_resolve_boundaries_no_trimming_fallback():
    cfg = _make_cfg()
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (None, None)
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
        )
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(0.0)
    assert outro_start == pytest.approx(2700.0)


def test_resolve_boundaries_preserves_zero_fingerprint_intro():
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (0.0, 0.0, 2550.0)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    mock_sil.assert_not_called()
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(0.0)
    assert outro_start == pytest.approx(2550.0)


def test_resolve_boundaries_fingerprint_none_falls_back_to_silence():
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (None, None, None)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (18.0, 2600.0)
        intro_start, intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    mock_sil.assert_called_once()
    assert intro_start == pytest.approx(0.0)
    assert intro_end == pytest.approx(18.0)
    assert outro_start == pytest.approx(2600.0)


# ---------------------------------------------------------------------------
# Chromaprint helpers (unit tests)
# ---------------------------------------------------------------------------


def test_popcount32_basic():
    """popcount of 0 is 0, popcount of 0xFFFFFFFF is 32."""
    arr = np.array([0, 0xFFFFFFFF, 1, 0b10101010], dtype=np.uint32)
    result = _popcount32(arr)
    np.testing.assert_array_equal(result, [0, 32, 1, 4])


def test_create_inverted_index():
    """The index maps each point to the LAST position it appeared."""
    fp = np.array([10, 20, 30, 10, 40], dtype=np.uint32)
    idx = _create_inverted_index(fp)
    assert idx[10] == 3   # last occurrence
    assert idx[20] == 1
    assert idx[30] == 2
    assert idx[40] == 4


def test_find_contiguous_range_single_block():
    times = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    r = _find_contiguous_range(times, max_gap=1.0)
    assert r is not None
    assert r[0] == pytest.approx(0.0)
    assert r[1] == pytest.approx(2.5)


def test_find_contiguous_range_two_blocks():
    """Should return the longer block."""
    times = np.array([0.0, 0.5, 1.0, 10.0, 10.5, 11.0, 11.5, 12.0])
    r = _find_contiguous_range(times, max_gap=1.0)
    assert r is not None
    assert r[0] == pytest.approx(10.0)
    assert r[1] == pytest.approx(12.0)


def test_find_contiguous_range_empty():
    times = np.array([], dtype=np.float64)
    assert _find_contiguous_range(times, max_gap=1.0) is None


def test_compare_episodes_chromaprint_identical_intro():
    """Two identical fingerprints should match the full length."""
    rng = np.random.RandomState(42)
    intro = rng.randint(0, 2**32, size=400, dtype=np.uint32)  # ~50s intro
    # Pad with unique filler so only the intro region matches.
    filler_a = rng.randint(0, 2**32, size=800, dtype=np.uint32)
    filler_b = rng.randint(0, 2**32, size=800, dtype=np.uint32)
    fp_a = np.concatenate([intro, filler_a])
    fp_b = np.concatenate([intro, filler_b])

    lr, rr = _compare_episodes_chromaprint(fp_a, fp_b, min_dur=5.0, max_dur=120.0)
    assert lr is not None and rr is not None
    # Both should start near 0 and end near 400 * 0.1238 ≈ 49.5 s
    assert lr[0] == pytest.approx(0.0, abs=1.0)
    assert lr[1] == pytest.approx(400 * 0.1238, abs=2.0)
    assert rr[0] == pytest.approx(0.0, abs=1.0)


def test_compare_episodes_chromaprint_shifted_intro():
    """Shared intro at different offsets (cold open)."""
    rng = np.random.RandomState(42)
    intro = rng.randint(0, 2**32, size=300, dtype=np.uint32)  # ~37s
    cold_open_a = rng.randint(0, 2**32, size=100, dtype=np.uint32)
    cold_open_b = rng.randint(0, 2**32, size=200, dtype=np.uint32)
    tail_a = rng.randint(0, 2**32, size=600, dtype=np.uint32)
    tail_b = rng.randint(0, 2**32, size=500, dtype=np.uint32)
    fp_a = np.concatenate([cold_open_a, intro, tail_a])
    fp_b = np.concatenate([cold_open_b, intro, tail_b])

    lr, rr = _compare_episodes_chromaprint(fp_a, fp_b, min_dur=5.0, max_dur=120.0)
    assert lr is not None and rr is not None
    # LHS should start at ~100 * 0.1238 ≈ 12.4s
    expected_start_a = 100 * 0.1238
    expected_start_b = 200 * 0.1238
    assert lr[0] == pytest.approx(expected_start_a, abs=2.0)
    assert rr[0] == pytest.approx(expected_start_b, abs=2.0)


def test_compare_episodes_chromaprint_no_common():
    """Completely different fingerprints should return None."""
    rng = np.random.RandomState(42)
    fp_a = rng.randint(0, 2**32, size=500, dtype=np.uint32)
    # Use a different seed to ensure no overlap
    rng2 = np.random.RandomState(99)
    fp_b = rng2.randint(0, 2**32, size=500, dtype=np.uint32)
    lr, rr = _compare_episodes_chromaprint(fp_a, fp_b, min_dur=15.0, max_dur=120.0)
    assert lr is None and rr is None


def test_compare_episodes_chromaprint_too_short():
    """A shared section shorter than min_dur should be rejected."""
    rng = np.random.RandomState(42)
    intro = rng.randint(0, 2**32, size=50, dtype=np.uint32)  # ~6s
    filler_a = rng.randint(0, 2**32, size=500, dtype=np.uint32)
    filler_b = rng.randint(0, 2**32, size=500, dtype=np.uint32)
    fp_a = np.concatenate([intro, filler_a])
    fp_b = np.concatenate([intro, filler_b])
    lr, rr = _compare_episodes_chromaprint(fp_a, fp_b, min_dur=15.0, max_dur=120.0)
    assert lr is None and rr is None


def test_pairwise_matches_returns_every_match():
    fps = {
        "a": np.array([1], dtype=np.uint32),
        "b": np.array([2], dtype=np.uint32),
        "c": np.array([3], dtype=np.uint32),
    }
    names = {1: "a", 2: "b", 3: "c"}
    table = {
        ("a", "b"): ((0.0, 20.0), (0.0, 20.0)),
        ("a", "c"): ((0.0, 30.0), (0.0, 30.0)),
        ("b", "c"): (None, None),
    }

    def _fake(lhs, rhs, min_dur, max_dur):
        return table[(names[int(lhs[0])], names[int(rhs[0])])]

    with patch("tvshow_noisemaker.detector._compare_episodes_chromaprint", side_effect=_fake):
        matches = _pairwise_matches(fps, ["a", "b", "c"], 15.0, 120.0, max_workers=2)

    assert matches == [
        ("a", (0.0, 20.0), "b", (0.0, 20.0)),
        ("a", (0.0, 30.0), "c", (0.0, 30.0)),
    ]


def _theme(a, b, start_a=0.0, start_b=0.0, length=30.0):
    return (a, (start_a, start_a + length), b, (start_b, start_b + length))


def test_consensus_prefers_the_length_most_episodes_share_over_a_longer_recap():
    """a–b share a 45 s recap (longest match of that pair); everyone shares a 30 s theme."""
    matches = [
        _theme("a", "b", length=45.0),  # recap: the a–b pair's longest match hides the theme
        _theme("a", "c"), _theme("a", "d"), _theme("b", "c"), _theme("b", "d"), _theme("c", "d"),
    ]
    c = _consensus_segments(matches, n_episodes=4)
    assert c.median_len == 30.0
    assert c.segments == {"a": (0.0, 30.0), "b": (0.0, 30.0), "c": (0.0, 30.0), "d": (0.0, 30.0)}
    assert c.support == {"a": 2, "b": 2, "c": 3, "d": 3}
    assert c.outliers == {"a": [45.0], "b": [45.0]}


def test_consensus_uses_median_of_an_episodes_cluster_matches():
    matches = [_theme("a", "b", start_a=10.0), _theme("a", "c", start_a=12.0), _theme("a", "d", start_a=11.0), _theme("b", "c"), _theme("b", "d"), _theme("c", "d")]
    c = _consensus_segments(matches, n_episodes=4)
    assert c.segments["a"] == (11.0, 41.0)


def test_consensus_tolerance_groups_lengths_within_three_seconds():
    matches = [_theme("a", "b", length=30.0), _theme("a", "c", length=32.5), _theme("b", "c", length=27.5)]
    c = _consensus_segments(matches, n_episodes=3)
    assert c.support == {"a": 2, "b": 2, "c": 2}
    assert c.outliers == {}


def test_consensus_run_span_is_capped_at_twice_the_tolerance():
    # 20..32 chained by 3 s steps would single-link into one 12 s run; cap splits it.
    matches = [_theme("a", "b", length=20.0), _theme("c", "d", length=23.0),
               _theme("e", "f", length=26.0), _theme("g", "h", length=29.0),
               _theme("i", "j", length=32.0)]
    c = _consensus_segments(matches, n_episodes=10)
    assert max(c.support.values()) == 1
    assert len(c.segments) <= 6  # a run covers at most 3 lengths (span 6 s)


def test_consensus_tie_goes_to_the_longer_cluster():
    matches = [_theme("a", "b", length=20.0), _theme("c", "d", length=40.0)]
    c = _consensus_segments(matches, n_episodes=4)
    assert c.median_len == 40.0
    assert set(c.segments) == {"c", "d"}
    assert c.outliers == {"a": [20.0], "b": [20.0]}


def test_consensus_needs_a_third_of_the_season():
    """One pair in a nine-episode season is not a season intro (needs max(2, ceil(9/3)) = 3 episodes)."""
    matches = [_theme("a", "b", length=40.0)]
    c = _consensus_segments(matches, n_episodes=9)
    assert c.median_len is None
    assert c.segments == {} and c.support == {}
    assert c.outliers == {"a": [40.0], "b": [40.0]}


def test_consensus_empty():
    c = _consensus_segments([], n_episodes=5)
    assert c.median_len is None and c.segments == {} and c.outliers == {}


def test_find_contiguous_at_shift_matching():
    """Identical arrays at shift=0 should yield a full-length match."""
    fp = np.array([100, 200, 300, 400, 500], dtype=np.uint32)
    lr, rr = _find_contiguous_at_shift(fp, fp, shift=0)
    assert lr is not None and rr is not None
    assert lr[1] > lr[0]


def test_find_contiguous_at_shift_no_match():
    """Completely different arrays should yield no match."""
    a = np.array([0x00000000] * 100, dtype=np.uint32)
    b = np.array([0xFFFFFFFF] * 100, dtype=np.uint32)
    lr, rr = _find_contiguous_at_shift(a, b, shift=0)
    assert lr is None and rr is None


def test_detect_by_chromaprint_parallel_extraction_deterministic_order():
    """Chromaprint extraction should run concurrently and preserve filepath order."""
    filepaths = [f"/show/ep{i:02d}.mp4" for i in range(1, 5)]
    durations = {fp: 1800.0 for fp in filepaths}

    lock = threading.Lock()
    active = 0
    max_active = 0

    def _fake_extract(_fp: str, _start: float, _duration: float):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return np.arange(200, dtype=np.uint32), False

    with (
        patch("tvshow_noisemaker.detector._extract_chromaprint", side_effect=_fake_extract),
        patch(
            "tvshow_noisemaker.detector._compare_episodes_chromaprint",
            return_value=((0.0, 30.0), (0.0, 30.0)),
        ),
    ):
        results = _detect_by_chromaprint(filepaths, durations)

    assert max_active > 1
    assert list(results.keys()) == filepaths
    assert results[filepaths[0]] == (0.0, 30.0, 1800.0 - 30.0)
    consensus = get_last_consensus()
    assert consensus.intro_len == 30.0 and consensus.outro_len == 30.0
    assert consensus.intro_episodes == 4 and consensus.outro_episodes == 4
    assert consensus.episodes[filepaths[0]] == MatchInfo(
        intro_support=3, outro_support=3, intro_len=30.0, outro_len=30.0, intro_outliers=[], outro_outliers=[],
    )


def test_detect_by_chromaprint_does_not_fill_unmatched_episodes():
    filepaths = [f"/show/ep{i:02d}.mp4" for i in range(1, 4)]
    durations = {fp: 1800.0 for fp in filepaths}

    def _fake_extract(fp, start, _duration):
        # Intro fingerprints are 1/2/3, outro fingerprints 101/102/103, so the fake
        # compare can tell passes and episodes apart without depending on call order.
        ep = int(fp[-6:-4])
        return np.full(200, ep if start < 0.5 else 100 + ep, dtype=np.uint32), False

    def _fake_compare(lhs, rhs, min_dur, max_dur):
        # Only ep01–ep02 share an intro; nothing shares credits.
        return ((0.0, 30.0), (0.0, 30.0)) if (int(lhs[0]), int(rhs[0])) == (1, 2) else (None, None)

    with (
        patch("tvshow_noisemaker.detector._extract_chromaprint", side_effect=_fake_extract),
        patch("tvshow_noisemaker.detector._compare_episodes_chromaprint", side_effect=_fake_compare),
    ):
        results = _detect_by_chromaprint(filepaths, durations)

    assert results[filepaths[0]] == (0.0, 30.0, None)
    assert results[filepaths[1]] == (0.0, 30.0, None)
    assert results[filepaths[2]] == (None, None, None)          # present, not median-filled
    c = get_last_consensus()
    assert c.intro_len == 30.0 and c.outro_len is None
    assert c.episodes[filepaths[2]].intro_support == 0


# ---------------------------------------------------------------------------
# _snap_to_silence / _refine_results_with_silence
# ---------------------------------------------------------------------------


def test_snap_to_silence_snaps_to_nearest_start():
    with patch(
        "tvshow_noisemaker.detector._run_silencedetect",
        return_value=([95.5, 99.8], [96.0, 100.2]),
    ):
        assert _snap_to_silence("f.mkv", 100.0, 1200.0) == 99.8


def test_snap_to_silence_returns_t_when_silencedetect_fails():
    with patch("tvshow_noisemaker.detector._run_silencedetect", return_value=None):
        assert _snap_to_silence("f.mkv", 100.0, 1200.0) == 100.0


def test_snap_to_silence_returns_t_when_no_silence_found():
    with patch("tvshow_noisemaker.detector._run_silencedetect", return_value=([], [])):
        assert _snap_to_silence("f.mkv", 100.0, 1200.0) == 100.0


def test_snap_to_silence_returns_t_near_file_start():
    with patch("tvshow_noisemaker.detector._run_silencedetect") as mock_sil:
        assert _snap_to_silence("f.mkv", 2.0, 1200.0) == 2.0
    mock_sil.assert_not_called()


def test_snap_to_silence_returns_t_near_file_end():
    with patch("tvshow_noisemaker.detector._run_silencedetect") as mock_sil:
        assert _snap_to_silence("f.mkv", 1199.0, 1200.0) == 1199.0
    mock_sil.assert_not_called()


def test_refine_results_with_silence_keeps_original_on_bad_ordering():
    """If snapping would push intro_start past intro_end, keep the original tuple."""
    results = {"/show/ep01.mp4": (10.0, 30.0, 2550.0)}
    durations = {"/show/ep01.mp4": 2700.0}

    def _fake_snap(_fp, t, _duration):
        if t == 10.0:
            return 35.0  # now >= intro_end, breaks ordering
        return t

    with patch(
        "tvshow_noisemaker.detector._snap_to_silence", side_effect=_fake_snap,
    ):
        refined = _refine_results_with_silence(results, durations)

    assert refined["/show/ep01.mp4"] == (10.0, 30.0, 2550.0)


def test_refine_results_with_silence_refines_normal_tuple():
    results = {
        "/show/ep01.mp4": (10.0, 30.0, 2550.0),
        "/show/ep02.mp4": (12.0, 32.0, 2560.0),
    }
    durations = {
        "/show/ep01.mp4": 2700.0,
        "/show/ep02.mp4": 2700.0,
        # ep03 has no duration and no entry in results anyway.
    }

    shifts = {10.0: 9.5, 30.0: 29.8, 2550.0: 2549.0, 12.0: 11.5, 32.0: 31.8, 2560.0: 2559.0}

    def _fake_snap(_fp, t, _duration):
        return shifts.get(t, t)

    with patch(
        "tvshow_noisemaker.detector._snap_to_silence", side_effect=_fake_snap,
    ):
        refined = _refine_results_with_silence(results, durations)

    assert refined["/show/ep01.mp4"] == (9.5, 29.8, 2549.0)
    assert refined["/show/ep02.mp4"] == (11.5, 31.8, 2559.0)


def test_refine_results_with_silence_passes_through_missing_duration():
    results = {
        "/show/ep01.mp4": (10.0, 30.0, 2550.0),
        "/show/ep02.mp4": (12.0, 32.0, 2560.0),
    }
    durations = {"/show/ep01.mp4": 2700.0}  # ep02 missing

    with patch(
        "tvshow_noisemaker.detector._snap_to_silence",
        side_effect=lambda _fp, t, _duration: t + 0.1,
    ):
        refined = _refine_results_with_silence(results, durations)

    assert refined["/show/ep01.mp4"] == (10.1, 30.1, 2550.1)
    assert refined["/show/ep02.mp4"] == (12.0, 32.0, 2560.0)
