"""Tests for tvshow_noisemaker.detector."""

import json
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

from tvshow_noisemaker.detector import (
    _coalesce,
    _find_alignment_offset,
    _find_common_segment_end,
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


def test_find_common_segment_end_no_cold_open():
    """When the theme starts at frame 0 in all episodes, should still work."""
    rng = np.random.RandomState(42)
    theme = rng.rand(12, 100).astype(np.float32) + 0.5

    fps = []
    for _ in range(3):
        fp = rng.rand(12, 300).astype(np.float32) * 0.1
        fp[:, :100] = theme + rng.randn(12, 100).astype(np.float32) * 0.02
        fps.append(fp)

    result = _find_common_segment_end(fps, sr=11025, min_duration=1.0, max_duration=60.0)
    assert result is not None
    # Theme ends near frame 100 → ~4.6s at 11025 Hz / 512 hop
    hop_sec = 512 / 11025
    expected = 100 * hop_sec
    for val in result:
        assert abs(val - expected) < 3.0, f"Expected ~{expected:.1f}s, got {val:.1f}s"


def test_find_common_segment_end_with_cold_open():
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

    result = _find_common_segment_end([fp1, fp2, fp3], sr=11025, min_duration=1.0, max_duration=60.0)
    assert result is not None
    assert len(result) == 3

    hop_sec = 512 / 11025
    # Each episode should have intro_end near where its theme ends
    expected = [130 * hop_sec, 160 * hop_sec, 110 * hop_sec]
    for got, exp in zip(result, expected):
        assert abs(got - exp) < 3.0, f"Expected ~{exp:.1f}s, got {got:.1f}s"


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
    )
    defaults.update(kwargs)
    return Config(**defaults)


def test_resolve_boundaries_uses_config_timestamps():
    cfg = _make_cfg(timestamps={"ep01.mp4": {"intro_end": 30.0, "outro_start": 2600.0}})
    intro_end, outro_start = resolve_boundaries(
        "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
    )
    assert intro_end == pytest.approx(30.0)
    assert outro_start == pytest.approx(2600.0)


def test_resolve_boundaries_prefers_timestamps_over_fingerprinting():
    cfg = _make_cfg(timestamps={"ep01.mp4": {"intro_end": 10.0, "outro_start": 2000.0}})
    fp_results = {"/show/ep01.mp4": (99.0, 500.0)}  # should be ignored
    intro_end, outro_start = resolve_boundaries(
        "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
    )
    assert intro_end == pytest.approx(10.0)
    assert outro_start == pytest.approx(2000.0)


def test_resolve_boundaries_uses_fingerprinting():
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (30.0, 2550.0)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    # Silence detection should NOT be called when fingerprinting covers both boundaries
    mock_sil.assert_not_called()
    assert intro_end == pytest.approx(30.0)
    assert outro_start == pytest.approx(2550.0)


def test_resolve_boundaries_falls_back_to_silence():
    cfg = _make_cfg()
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (28.0, 2580.0)
        intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
        )
    mock_sil.assert_called_once()
    assert intro_end == pytest.approx(28.0)
    assert outro_start == pytest.approx(2580.0)


def test_resolve_boundaries_falls_back_to_user_duration():
    cfg = _make_cfg(intro_duration=60.0, outro_duration=30.0)
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (None, None)
        intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
        )
    assert intro_end == pytest.approx(60.0)
    assert outro_start == pytest.approx(2700.0 - 30.0)


def test_resolve_boundaries_no_trimming_fallback():
    cfg = _make_cfg()
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (None, None)
        intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results={}, duration=2700.0
        )
    assert intro_end == pytest.approx(0.0)
    assert outro_start == pytest.approx(2700.0)


def test_resolve_boundaries_preserves_zero_fingerprint_intro():
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (0.0, 2550.0)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    mock_sil.assert_not_called()
    assert intro_end == pytest.approx(0.0)
    assert outro_start == pytest.approx(2550.0)


def test_resolve_boundaries_fingerprint_none_falls_back_to_silence():
    cfg = _make_cfg()
    fp_results = {"/show/ep01.mp4": (None, None)}
    with patch("tvshow_noisemaker.detector.detect_by_silence") as mock_sil:
        mock_sil.return_value = (18.0, 2600.0)
        intro_end, outro_start = resolve_boundaries(
            "/show/ep01.mp4", cfg=cfg, fingerprint_results=fp_results, duration=2700.0
        )
    mock_sil.assert_called_once()
    assert intro_end == pytest.approx(18.0)
    assert outro_start == pytest.approx(2600.0)
