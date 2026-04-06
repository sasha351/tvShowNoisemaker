"""Tests for tvshow_noisemaker.detector."""

import json
import threading
import time
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from tvshow_noisemaker.detector import (
    _coalesce,
    _detect_by_chromaprint,
    _compare_episodes_chromaprint,
    _create_inverted_index,
    _find_alignment_offset,
    _find_common_segment_bounds,
    _find_contiguous_at_shift,
    _find_contiguous_range,
    _match_template_position,
    _otsu_threshold,
    _popcount32,
    _frame_similarity,
    _smooth,
    detect_by_silence,
    get_duration,
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
    starts, ends, seg_start, seg_len = result
    # Theme ends near frame 100 → ~4.6s at 11025 Hz / 512 hop
    hop_sec = 512 / 11025
    expected_end = 100 * hop_sec
    for val in ends:
        assert abs(val - expected_end) < 3.0, f"Expected ~{expected_end:.1f}s, got {val:.1f}s"
    # Theme starts at frame 0 → starts should be near 0
    for val in starts:
        assert val < 3.0, f"Expected start near 0s, got {val:.1f}s"


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
    starts, ends, seg_start, seg_len = result
    assert len(ends) == 3
    assert len(starts) == 3

    hop_sec = 512 / 11025
    # Each episode should have intro_end near where its theme ends
    expected_ends = [130 * hop_sec, 160 * hop_sec, 110 * hop_sec]
    for got, exp in zip(ends, expected_ends):
        assert abs(got - exp) < 3.0, f"Expected end ~{exp:.1f}s, got {got:.1f}s"
    # Each episode should have intro_start near where its theme starts
    expected_starts = [50 * hop_sec, 80 * hop_sec, 30 * hop_sec]
    for got, exp in zip(starts, expected_starts):
        assert abs(got - exp) < 3.0, f"Expected start ~{exp:.1f}s, got {got:.1f}s"


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
    _, _, seg_start, seg_len = result
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
    _, _, seg_start, seg_len = result
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


# ---------------------------------------------------------------------------
# detect_by_silence
# ---------------------------------------------------------------------------

_SILENCE_STDERR = """\
[silencedetect @ 0x...] silence_start: 0
[silencedetect @ 0x...] silence_end: 28.5 | silence_duration: 28.5
[silencedetect @ 0x...] silence_start: 2550.0
[silencedetect @ 0x...] silence_end: 2600.0 | silence_duration: 50.0
"""


def test_detect_by_silence_parses_output():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="",
            stderr=_SILENCE_STDERR,
        )
        intro_end, outro_start = detect_by_silence(
            "/fake/ep01.mp4", duration=2700.0
        )
    assert intro_end == pytest.approx(28.5)
    assert outro_start == pytest.approx(2550.0)


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
    """If the last silence is in the first 50% of the file it should be ignored."""
    stderr = (
        "[silencedetect] silence_start: 0\n"
        "[silencedetect] silence_end: 10.0 | silence_duration: 10.0\n"
        "[silencedetect] silence_start: 100.0\n"  # < 50% of 2700
        "[silencedetect] silence_end: 101.0 | silence_duration: 1.0\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr=stderr)
        _, outro_start = detect_by_silence("/fake/ep01.mp4", duration=2700.0)
    assert outro_start is None


def test_detect_by_silence_caches_results(tmp_path):
    """Second call with same file should use cache, not ffmpeg."""
    fake_file = tmp_path / "ep01.mp4"
    fake_file.write_bytes(b"\x00" * 100)

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="", stderr=_SILENCE_STDERR,
        )
        r1 = detect_by_silence(str(fake_file), duration=2700.0)

    assert mock_run.call_count == 1
    assert r1 == (pytest.approx(28.5), pytest.approx(2550.0))

    # Second call should hit cache, not call subprocess
    with patch("subprocess.run") as mock_run2:
        r2 = detect_by_silence(str(fake_file), duration=2700.0)

    mock_run2.assert_not_called()
    assert r2 == r1


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
