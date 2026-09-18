"""Intro/outro boundary detection for TV episodes.

Detection priority (per batch):
1. Audio fingerprinting over the first 5 episodes (librosa chroma features).
    Preferred because it finds identical/near-identical segments shared
    by episodes in the batch, and runs in parallel with on-disk caching.
2. Silence detection (ffmpeg ``silencedetect`` filter).
   Works well when intros/outros are separated from content by silence.
3. User-supplied timestamps/durations from config or CLI flags.
   Always used when fingerprinting and silence detection both fail, or
   when the user explicitly provides values.

Tradeoffs
---------
* **Fingerprinting** is the most accurate method for TV shows where
    the intro/outro are identical across episodes. It uses up to 5 files
    for speed, and it decodes audio/computes chroma features in parallel.
* **Silence detection** is fast but unreliable for compressed audio or
  episodes without a clear silence gap around the intro/outro.
* **User timestamps** are always reliable but require manual inspection
  of at least one episode.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import math
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import librosa as _librosa
except ImportError:  # pragma: no cover
    _librosa = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Sample rate used when extracting audio for fingerprinting.
# 11 025 Hz is sufficient for chroma analysis and much faster to decode.
_FINGERPRINT_SR = 11_025
_HOP_LENGTH = 512  # frames per chroma window
_MAX_FINGERPRINT_FILES = 5
_CACHE_VERSION = "v1"
_CACHE_DIR = Path.home() / ".cache" / "tvshow_noisemaker" / "fingerprints"
_SILENCE_CACHE_DIR = Path.home() / ".cache" / "tvshow_noisemaker" / "silence"
_CHROMAPRINT_CACHE_DIR = Path.home() / ".cache" / "tvshow_noisemaker" / "chromaprint"

# Silence detection scan windows: the intro boundary is only searched in the
# first 5 minutes and the outro near the end, so decode only those regions.
_SILENCE_INTRO_WINDOW = 330.0
_SILENCE_OUTRO_WINDOW = 600.0

# Chromaprint constants (adapted from intro-skipper's detection algorithm).
# Each Chromaprint point represents ~0.1238 seconds of audio.
_CHROMAPRINT_SAMPLES_TO_SEC = 0.1238
# Max differing bits (out of 32) for two fingerprint points to be considered matching.
_CHROMAPRINT_MAX_DIFF = 6
# Max gap (seconds) allowed inside a contiguous matching region.
_CHROMAPRINT_MAX_TIME_SKIP = 3.5
# Fuzzy lookup range (±) when searching the inverted fingerprint index.
_CHROMAPRINT_INDEX_SHIFT = 2
# Popcount lookup table (byte → number of set bits), computed once at import.
_POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)
# Cached result of fpcalc availability check.
_fpcalc_available: Optional[bool] = None

# Silence-snap refinement (intro-skipper: AdjustWindow* / SilenceDetection*)
_REFINE_WINDOW_BEFORE = 5.0   # seconds searched before a detected boundary
_REFINE_WINDOW_AFTER = 2.0    # seconds searched after a detected boundary
_REFINE_NOISE_DB = -50.0
_REFINE_MIN_SILENCE = 0.33

# Cross-correlation refinement of Chromaprint boundaries against a reference episode.
_XCORR_SR = 8000
_XCORR_HOP = 200                 # 25 ms frames
_XCORR_NFFT = 1024
_XCORR_WINDOW = 6.0              # seconds on each side of the raw boundary
_XCORR_MAX_LAG = 2.0             # seconds; raw boundaries of two episodes disagree by less than this
_XCORR_SIM_THRESHOLD = 0.7       # ponytail: calibration knob — 0.7 errs toward keeping theme (measured on 3 shows)
_XCORR_SMOOTH = 8                # frames (0.2 s) of similarity smoothing
_XCORR_PROBE = 2.5               # seconds inside the shared region where similarity must already be high
_XCORR_SNAP_WINDOW = 1.0         # seconds; refined boundaries only snap to a silence this close, safe side only


@dataclass
class FingerprintStats:
    """Runtime statistics for the most recent fingerprinting pass."""

    elapsed_sec: float = 0.0
    candidate_files: int = 0
    jobs_submitted: int = 0
    valid_files: int = 0
    workers: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


_LAST_FINGERPRINT_STATS = FingerprintStats()


def _set_last_fingerprint_stats(stats: FingerprintStats) -> None:
    global _LAST_FINGERPRINT_STATS
    _LAST_FINGERPRINT_STATS = stats


def get_last_fingerprint_stats() -> Dict[str, float]:
    """Return runtime statistics from the last fingerprinting pass."""
    return asdict(_LAST_FINGERPRINT_STATS)


# ── Season consensus ────────────────────────────────────────────────────────

# ponytail: mkv chapter markers are not used to adjust boundaries
# (intro-skipper does); add when a show ships chapters that mark the intro.

_CONSENSUS_TOLERANCE = 3.0  # seconds; pair matches within this of a cluster centre vote together

PairMatch = Tuple[str, Tuple[float, float], str, Tuple[float, float]]


@dataclass
class MatchInfo:
    """How strongly one episode supports the season consensus."""

    intro_support: int = 0            # pair matches inside the winning intro cluster
    outro_support: int = 0
    intro_len: Optional[float] = None  # this episode's raw (pre-snap) matched intro length
    outro_len: Optional[float] = None
    intro_outliers: List[float] = field(default_factory=list)  # lengths of ignored intro matches (recaps…)
    outro_outliers: List[float] = field(default_factory=list)


@dataclass
class SeasonConsensus:
    """What the season agrees on: shared intro/credits length and who supports it."""

    intro_len: Optional[float] = None  # median length of the winning intro cluster; None = no shared intro
    outro_len: Optional[float] = None  # same for credits, measured from the end of the file
    intro_episodes: int = 0            # episodes covered by the winning cluster
    outro_episodes: int = 0
    episodes: Dict[str, MatchInfo] = field(default_factory=dict)
    intro_refs: List[str] = field(default_factory=list)  # top-2 episodes by intro support, for xcorr refinement
    outro_refs: List[str] = field(default_factory=list)


# ponytail: same module-level side channel as the fingerprint stats; the
# public detect_by_fingerprinting() return type stays a plain dict.
_last_consensus = SeasonConsensus()


def _set_last_consensus(consensus: SeasonConsensus) -> None:
    global _last_consensus
    _last_consensus = consensus


def get_last_consensus() -> SeasonConsensus:
    """Consensus published by the most recent Chromaprint pass."""
    return _last_consensus


@dataclass
class _Consensus:
    segments: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # per-episode segment in the winning cluster
    support: Dict[str, int] = field(default_factory=dict)                   # cluster matches per episode
    outliers: Dict[str, List[float]] = field(default_factory=dict)          # ignored match lengths per episode
    median_len: Optional[float] = None                                       # None = no consensus


def _consensus_segments(
    matches: List[PairMatch], n_episodes: int, tolerance: float = _CONSENSUS_TOLERANCE,
) -> _Consensus:
    """Pick the shared-segment length most episodes agree on, and each episode's segment in it.

    Every pair match votes once per side with its length.  Distinct lengths
    are grouped into runs by single-linkage: consecutive lengths (sorted)
    join the same run while the gap between them is within *tolerance*, so a
    run can span more than *tolerance* end-to-end when chained through an
    intermediate vote.  The winning run has the most pair-match votes (ties →
    longer) among runs covering at least ``max(2, ceil(n_episodes / 3))``
    episodes; with no such run the season has no consensus.  An episode's segment is the median of its
    run votes; votes outside the run (recaps, two-parters, shared cues) are
    returned as outlier lengths so callers can flag them.
    """
    out = _Consensus()
    votes = []  # (length, episode, segment)
    for a, seg_a, b, seg_b in matches:
        votes.append((seg_a[1] - seg_a[0], a, seg_a))
        votes.append((seg_b[1] - seg_b[0], b, seg_b))
    if not votes:
        return out

    # Group vote lengths into runs by single-linkage: sort the distinct
    # lengths and cut wherever the gap to the previous one exceeds
    # tolerance. A run can span more than `tolerance` end-to-end when
    # chained by an intermediate vote (e.g. 27.5–30–32.5 at tolerance=3);
    # a plain "within tolerance of one centre" scan under-covers that case.
    # ponytail: O(n log n) sort + linear scan, fine at season-sized vote counts.
    runs: List[List[float]] = []
    for length in sorted({v[0] for v in votes}):
        # ponytail: no cap on a run's total span — real credits votes spread 31–40 s
        # (variable tail matches) and capping split that cluster so a 20 s tail won.
        if runs and length - runs[-1][-1] <= tolerance:
            runs[-1].append(length)
        else:
            runs.append([length])

    # The winner is the run with the most pair-match votes (ties → longer), among
    # runs covering enough episodes.  Cover alone is wrong: a 20 s closing logo
    # shared by every episode beat 36 s credits that one episode lacked.
    min_cover = max(2, math.ceil(n_episodes / 3))
    best_run: List[float] = []
    best_votes = 0
    for run in sorted(runs, key=lambda r: r[-1], reverse=True):  # longer first so ties keep the longer
        run_set = set(run)
        cover = len({ep for length, ep, _ in votes if length in run_set})
        n_votes = sum(1 for length, _, _ in votes if length in run_set)
        if cover >= min_cover and n_votes > best_votes:
            best_run, best_votes = run, n_votes
    if not best_run:
        for length, ep, _ in votes:
            out.outliers.setdefault(ep, []).append(round(length, 1))
        return out

    best_set = set(best_run)
    by_ep: Dict[str, List[Tuple[float, float]]] = {}
    for length, ep, seg in votes:
        if length in best_set:
            by_ep.setdefault(ep, []).append(seg)
        else:
            out.outliers.setdefault(ep, []).append(round(length, 1))
    out.median_len = float(np.median([length for length, _, _ in votes if length in best_set]))
    for ep, segs in by_ep.items():
        out.segments[ep] = (float(np.median([s[0] for s in segs])), float(np.median([s[1] for s in segs])))
        out.support[ep] = len(segs)
    return out


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def get_duration(filepath: str, timeout: float = 30.0) -> Optional[float]:
    """Return the duration of *filepath* in seconds using ``ffprobe``.

    Returns ``None`` if the duration cannot be determined.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        filepath,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            logger.debug("ffprobe returned non-zero for %s: %s", filepath, result.stderr)
            return None
        info = json.loads(result.stdout)
        return float(info["format"]["duration"])
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("Failed to get duration for %s: %s", filepath, exc)
        return None


def _extract_audio_pcm(
    filepath: str,
    start: float,
    duration: float,
    sr: int = _FINGERPRINT_SR,
) -> Optional[np.ndarray]:
    """Decode a segment of *filepath* to a mono float32 numpy array.

    Uses ``ffmpeg`` raw PCM output piped to stdout so that librosa is only
    used for feature extraction (not for decoding), keeping the dependency
    surface minimal and avoiding heavy codec overhead from soundfile.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", filepath,
        "-ac", "1",
        "-ar", str(sr),
        "-f", "f32le",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode != 0 or not result.stdout:
            logger.debug("ffmpeg PCM extraction failed for %s", filepath)
            return None
        audio = np.frombuffer(result.stdout, dtype=np.float32)
        return audio
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("PCM extraction exception for %s: %s", filepath, exc)
        return None


def _chroma_fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute a chroma STFT fingerprint matrix (12 × n_frames)."""
    if _librosa is None:
        raise RuntimeError(
            "librosa is required for chroma STFT fingerprinting but is not installed. "
            "Install fpcalc (`brew install chromaprint`) for the primary detection path, "
            "run `pip install 'tvshow-noisemaker[stft]'` for this fallback, "
            "or use --no-fingerprint to disable fingerprinting."
        )

    if audio.size == 0:
        return np.zeros((12, 0), dtype=np.float32)
    return _librosa.feature.chroma_stft(
        y=audio, sr=sr, hop_length=_HOP_LENGTH
    ).astype(np.float32)


def _frame_similarity(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """Compute per-frame cosine similarity between two chroma matrices.

    Both matrices are 12 × n_frames.  The shorter one limits the output
    length.
    """
    n = min(fp_a.shape[1], fp_b.shape[1])
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    a = fp_a[:, :n]
    b = fp_b[:, :n]
    norm_a = np.linalg.norm(a, axis=0, keepdims=True) + 1e-8
    norm_b = np.linalg.norm(b, axis=0, keepdims=True) + 1e-8
    return ((a / norm_a) * (b / norm_b)).sum(axis=0)


def _smooth(arr: np.ndarray, window_frames: int) -> np.ndarray:
    """Box-car smooth *arr* with a window of *window_frames* frames."""
    if window_frames <= 1:
        return arr
    kernel = np.ones(window_frames, dtype=np.float32) / window_frames
    return np.convolve(arr, kernel, mode="same")


def _otsu_threshold(arr: np.ndarray) -> float:
    """Compute Otsu's threshold for a 1-D array of continuous values.

    Finds the value that best separates the distribution into two classes
    (e.g. "theme" vs "non-theme" similarity scores) by maximising the
    inter-class variance.  This is fully adaptive — it works regardless of
    the theme-to-episode length ratio or the absolute similarity scale.
    """
    n_bins = 256
    mn, mx = float(arr.min()), float(arr.max())
    if mx - mn < 1e-8:
        return float(mn)
    hist, bin_edges = np.histogram(arr, bins=n_bins, range=(mn, mx))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    total = hist.sum()
    if total == 0:
        return float(mn)

    sum_total = float((hist * bin_centers).sum())
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = float(mn)

    for i in range(n_bins):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += hist[i] * bin_centers[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = float(bin_centers[i])

    return best_threshold


def _next_power_of_two(n: int) -> int:
    """Return the smallest power-of-two FFT length >= *n*."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _find_alignment_offset(
    ref: np.ndarray,
    other: np.ndarray,
    max_lag_frames: int,
) -> int:
    """Return frame offset *d* that best aligns *other* with *ref*.

    Uses FFT cross-correlation across all 12 chroma bands.
    The result means ``other[:, i - d]`` corresponds to ``ref[:, i]``.
    Positive *d*: the common segment starts later in *other* than in *ref*.
    """
    n_ref = ref.shape[1]
    n_other = other.shape[1]
    if n_ref == 0 or n_other == 0:
        return 0

    linear_corr_len = n_ref + n_other - 1
    n_fft = _next_power_of_two(linear_corr_len)
    # Batch FFT across all 12 chroma bands at once
    fa = np.fft.rfft(ref, n=n_fft, axis=1)
    fb = np.fft.rfft(other, n=n_fft, axis=1)
    corr = np.fft.irfft(fa * np.conj(fb), n=n_fft, axis=1).sum(axis=0)

    # Only consider lags within ±max_lag_frames
    mask = np.full(n_fft, -np.inf)
    lag_limit = min(max_lag_frames + 1, linear_corr_len)
    indices = np.arange(lag_limit)
    mask[indices] = corr[indices]
    neg_indices = n_fft - indices[1:]  # skip k=0 (n_fft - 0 == n_fft, out of bounds)
    valid_neg = neg_indices[(neg_indices > 0) & (neg_indices < n_fft)]
    mask[valid_neg] = corr[valid_neg]

    peak = int(np.argmax(mask))
    lag = peak if peak <= n_fft // 2 else peak - n_fft
    return lag


def _find_common_segment_bounds(
    fingerprints: List[np.ndarray],
    sr: int,
    min_duration: float,
    max_duration: float,
) -> Optional[Tuple[int, int]]:
    """Find where a common segment starts and ends for each fingerprinted episode.

    Uses cross-correlation to align episodes (handles cold opens where the
    shared theme song starts at a different time in each episode), then
    finds the longest contiguous high-similarity run.

    Returns ``(seg_start_frame, seg_len_frames)`` in the coordinate system
    of the *reference* (first) fingerprint, or ``None`` if no common
    segment exceeding *min_duration* is found.  Callers locate per-episode
    boundaries by template-matching this segment against each episode.
    """
    hop_sec = _HOP_LENGTH / sr
    ref = fingerprints[0]
    n_ref = ref.shape[1]
    if n_ref == 0:
        return None

    max_lag = n_ref // 2

    # ── Step 1: align every fingerprint to the reference ────────
    lags = [0]
    for idx, other in enumerate(fingerprints[1:], 1):
        lag = _find_alignment_offset(ref, other, max_lag)
        lags.append(lag)
        logger.debug("  Alignment: episode %d lag = %d frames (%.1fs)", idx, lag, lag * hop_sec)

    # ── Step 2: compute aligned per-frame similarity ────────────
    scores = np.zeros(n_ref, dtype=np.float64)
    counts = np.zeros(n_ref, dtype=np.float64)

    for other, lag in zip(fingerprints[1:], lags[1:]):
        n_other = other.shape[1]
        # In the reference coordinate system, valid overlap is
        # ref[start:end] ↔ other[start-lag : end-lag]
        start = max(0, lag)
        end = min(n_ref, n_other + lag)
        if end - start < 1:
            continue

        ref_chunk = ref[:, start:end]
        other_chunk = other[:, start - lag : end - lag]
        sim = _frame_similarity(ref_chunk, other_chunk)
        scores[start:end] += sim
        counts[start:end] += 1

    valid = counts > 0
    scores[valid] /= counts[valid]
    scores = scores.astype(np.float32)

    # ── Step 3: smooth and threshold ────────────────────────────
    smooth_window = max(1, int(2.0 / hop_sec))
    scores = _smooth(scores, smooth_window)

    threshold = _otsu_threshold(scores[valid] if valid.any() else scores)
    min_frames = max(1, int(min_duration / hop_sec))
    max_frames = int(max_duration / hop_sec)

    # ── Step 4: longest run of above-threshold frames ───────────
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for i in range(n_ref):
        if scores[i] >= threshold:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_len = 0
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start

    if best_len < min_frames:
        logger.debug(
            "  No common segment: best_len=%d frames (%.1fs) < min=%d frames (%.1fs), "
            "threshold=%.4f, score range=[%.4f, %.4f]",
            best_len, best_len * hop_sec, min_frames, min_duration,
            threshold,
            float(scores[valid].min()) if valid.any() else 0,
            float(scores[valid].max()) if valid.any() else 0,
        )
        return None

    if best_len > max_frames:
        best_len = max_frames

    segment_end_ref = best_start + best_len
    logger.debug(
        "  Common segment: frames %d–%d (%.1fs–%.1fs), len=%d (%.1fs), threshold=%.4f",
        best_start, segment_end_ref,
        best_start * hop_sec, segment_end_ref * hop_sec,
        best_len, best_len * hop_sec, threshold,
    )

    return best_start, best_len


def _match_template_position(
    template: np.ndarray,
    episode_fp: np.ndarray,
    sr: int,
) -> Optional[Tuple[float, float]]:
    """Find where *template* best matches inside *episode_fp*.

    Uses FFT cross-correlation across all 12 chroma bands.

    Returns ``(start_seconds, end_seconds)`` or ``None`` if the template
    is longer than the episode fingerprint.
    """
    hop_sec = _HOP_LENGTH / sr
    T = template.shape[1]
    N = episode_fp.shape[1]
    if T == 0 or N == 0 or T > N:
        return None

    n_fft = _next_power_of_two(N + T - 1)
    fa = np.fft.rfft(episode_fp, n=n_fft, axis=1)
    fb = np.fft.rfft(template, n=n_fft, axis=1)
    corr = np.fft.irfft(fa * np.conj(fb), n=n_fft, axis=1).sum(axis=0)

    valid_len = N - T + 1
    peak_idx = int(np.argmax(corr[:valid_len]))

    start_time = peak_idx * hop_sec
    end_time = (peak_idx + T) * hop_sec
    return start_time, end_time


def _fingerprint_cache_path(
    filepath: str,
    segment: str,
    start: float,
    duration: float,
    sr: int,
) -> Path:
    """Return a stable cache path for a fingerprint segment."""
    try:
        st = os.stat(filepath)
        size = st.st_size
        mtime_ns = st.st_mtime_ns
    except OSError:
        size = -1
        mtime_ns = -1

    key = "|".join(
        (
            filepath,
            segment,
            f"{start:.3f}",
            f"{duration:.3f}",
            str(sr),
            str(_HOP_LENGTH),
            str(size),
            str(mtime_ns),
            _CACHE_VERSION,
        )
    )
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _CACHE_DIR / f"{digest}.npz"


def _load_cached_fingerprint(path: Path) -> Optional[np.ndarray]:
    """Load cached fingerprint if available and valid."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            fp = data["fp"]
            if fp.ndim != 2 or fp.shape[0] != 12:
                return None
            return fp.astype(np.float32, copy=False)
    except Exception:  # pylint: disable=broad-except
        return None


def _save_cached_fingerprint(path: Path, fp: np.ndarray) -> None:
    """Persist fingerprint to cache (best-effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fp=fp.astype(np.float32, copy=False))
    except Exception:  # pylint: disable=broad-except
        return


def _get_segment_fingerprint(
    filepath: str,
    segment: str,
    start: float,
    duration: float,
    sr: int,
) -> Tuple[Optional[np.ndarray], bool]:
    """Load a segment fingerprint from cache or compute and cache it."""
    cache_path = _fingerprint_cache_path(filepath, segment, start, duration, sr)
    cached = _load_cached_fingerprint(cache_path)
    if cached is not None:
        return cached, True

    audio = _extract_audio_pcm(filepath, start=start, duration=duration, sr=sr)
    if audio is None:
        return None, False

    fp = _chroma_fingerprint(audio, sr)
    _save_cached_fingerprint(cache_path, fp)
    return fp, False


def _compute_file_fingerprints(
    filepath: str,
    duration: float,
    search_window: float,
    sr: int,
) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray], int, int]:
    """Compute intro/outro fingerprints for one file (cache-aware)."""
    seg_duration = min(search_window, duration)
    intro_fp, intro_cached = _get_segment_fingerprint(
        filepath,
        segment="intro",
        start=0.0,
        duration=seg_duration,
        sr=sr,
    )
    outro_fp, outro_cached = _get_segment_fingerprint(
        filepath,
        segment="outro",
        start=max(0.0, duration - search_window),
        duration=seg_duration,
        sr=sr,
    )
    hits = int(intro_cached) + int(outro_cached)
    misses = int(not intro_cached) + int(not outro_cached)
    return filepath, intro_fp, outro_fp, hits, misses


# ---------------------------------------------------------------------------
# Chromaprint-based detection (primary, adapted from intro-skipper)
# ---------------------------------------------------------------------------


def _check_fpcalc() -> bool:
    """Return True if the ``fpcalc`` CLI tool (Chromaprint) is available."""
    global _fpcalc_available
    if _fpcalc_available is not None:
        return _fpcalc_available
    try:
        result = subprocess.run(
            ["fpcalc", "-version"], capture_output=True, text=True, timeout=5
        )
        _fpcalc_available = result.returncode == 0
    except Exception:  # pylint: disable=broad-except
        _fpcalc_available = False
    if _fpcalc_available:
        logger.info("fpcalc (Chromaprint) detected – using Chromaprint fingerprinting")
    else:
        logger.info("fpcalc not available – falling back to chroma STFT fingerprinting")
    return _fpcalc_available


def _chromaprint_cache_path(
    filepath: str, start: float, duration: float,
) -> Path:
    """Return a stable cache path for a Chromaprint fingerprint."""
    try:
        st = os.stat(filepath)
        size, mtime_ns = st.st_size, st.st_mtime_ns
    except OSError:
        size, mtime_ns = -1, -1
    key = "|".join((
        filepath, f"{start:.3f}", f"{duration:.3f}",
        "chromaprint", str(size), str(mtime_ns), _CACHE_VERSION,
    ))
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _CHROMAPRINT_CACHE_DIR / f"{digest}.npy"


def _fpcalc_direct(filepath: str, max_length: float) -> Optional[np.ndarray]:
    """Run ``fpcalc`` directly on *filepath* (starts from the beginning)."""
    cmd = [
        "fpcalc", "-raw", "-json",
        "-length", str(int(max_length + 1)),
        filepath,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.debug("fpcalc failed for %s: %s", filepath, result.stderr[:200])
            return None
        data = json.loads(result.stdout)
        fp = data.get("fingerprint", [])
        return np.array(fp, dtype=np.uint32) if fp else None
    except Exception as exc:
        logger.debug("fpcalc exception for %s: %s", filepath, exc)
        return None


def _fpcalc_segment(
    filepath: str, start: float, duration: float,
) -> Optional[np.ndarray]:
    """Extract an audio segment with ffmpeg, then fingerprint it with ``fpcalc``."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(duration),
            "-i", filepath, "-ac", "1", "-ar", "16000",
            "-f", "wav", tmp_path,
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=120)
        if r.returncode != 0:
            return None
        cmd = [
            "fpcalc", "-raw", "-json",
            "-length", str(int(duration + 1)),
            tmp_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        data = json.loads(r.stdout)
        fp = data.get("fingerprint", [])
        return np.array(fp, dtype=np.uint32) if fp else None
    except Exception as exc:
        logger.debug("fpcalc segment exception for %s: %s", filepath, exc)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _extract_chromaprint(
    filepath: str, start: float, duration: float,
) -> Tuple[Optional[np.ndarray], bool]:
    """Extract a Chromaprint fingerprint from the given segment.

    Returns ``(fingerprint_array, was_cached)``.
    """
    cache_path = _chromaprint_cache_path(filepath, start, duration)
    if cache_path.exists():
        try:
            fp = np.load(cache_path, allow_pickle=False)
            if fp.dtype == np.uint32 and fp.ndim == 1 and len(fp) > 0:
                return fp, True
        except Exception:
            pass

    fp = _fpcalc_direct(filepath, duration) if start < 0.5 else _fpcalc_segment(filepath, start, duration)

    if fp is not None and len(fp) > 0:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, fp)
        except Exception:
            pass
        return fp, False
    return None, False


# ── Chromaprint comparison helpers ──────────────────────────────────────────


def _popcount32(arr: np.ndarray) -> np.ndarray:
    """Count set bits for each uint32 element (vectorised via byte LUT)."""
    return _POPCOUNT_LUT[arr.view(np.uint8)].reshape(-1, 4).sum(axis=1)


def _create_inverted_index(fp: np.ndarray) -> Dict[int, int]:
    """Map each fingerprint point value to the *last* index it appeared at."""
    return dict(zip(fp.tolist(), range(len(fp))))


def _find_contiguous_range(
    times: np.ndarray, max_gap: float,
) -> Optional[Tuple[float, float]]:
    """Return the start and end of longest contiguous block (gap ≤ *max_gap*)."""
    if len(times) == 0:
        return None
    best_s = best_e = cur_s = cur_e = times[0]
    best_d = 0.0
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] <= max_gap:
            cur_e = times[i + 1]
        else:
            d = cur_e - cur_s
            if d > best_d:
                best_d, best_s, best_e = d, cur_s, cur_e
            cur_s = cur_e = times[i + 1]
    d = cur_e - cur_s
    if d > best_d:
        best_d, best_s, best_e = d, cur_s, cur_e
    return (float(best_s), float(best_e)) if best_d > 0 else None


def _find_contiguous_at_shift(
    lhs: np.ndarray, rhs: np.ndarray, shift: int,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """XOR-compare fingerprints at the given *shift* and find matching runs."""
    left_off = max(0, -shift)
    right_off = max(0, shift)
    upper = min(len(lhs), len(rhs)) - abs(shift)
    if upper <= 0:
        return None, None

    diff = np.bitwise_xor(
        lhs[left_off : left_off + upper],
        rhs[right_off : right_off + upper],
    )
    bits = _popcount32(diff)
    matches = np.where(bits <= _CHROMAPRINT_MAX_DIFF)[0]
    if len(matches) == 0:
        return None, None

    lhs_times = np.sort((matches + left_off).astype(np.float64) * _CHROMAPRINT_SAMPLES_TO_SEC)
    rhs_times = np.sort((matches + right_off).astype(np.float64) * _CHROMAPRINT_SAMPLES_TO_SEC)

    lhs_range = _find_contiguous_range(lhs_times, _CHROMAPRINT_MAX_TIME_SKIP)
    if lhs_range is None:
        return None, None
    rhs_range = _find_contiguous_range(rhs_times, _CHROMAPRINT_MAX_TIME_SKIP)
    return lhs_range, rhs_range


def _compare_episodes_chromaprint(
    lhs_fp: np.ndarray,
    rhs_fp: np.ndarray,
    min_dur: float = 15.0,
    max_dur: float = 120.0,
) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    """Compare two Chromaprint fingerprints using inverted-index shift search.

    Algorithm based on intro-skipper:
    1. Build inverted indexes (fingerprint value → last position).
    2. Find alignment shifts where matching points exist.
    3. For each shift, XOR + popcount to find similar regions.
    4. Pick the longest contiguous region within [min_dur, max_dur].

    Returns ``((lhs_start, lhs_end), (rhs_start, rhs_end))`` or ``(None, None)``.
    """
    if len(lhs_fp) == 0 or len(rhs_fp) == 0:
        return None, None

    lhs_idx = _create_inverted_index(lhs_fp)
    rhs_idx = _create_inverted_index(rhs_fp)

    # Discover candidate alignment shifts via fuzzy index lookup.
    shifts: set = set()
    for point, lhs_pos in lhs_idx.items():
        for off in range(-_CHROMAPRINT_INDEX_SHIFT, _CHROMAPRINT_INDEX_SHIFT + 1):
            mod = (point + off) & 0xFFFFFFFF
            if mod in rhs_idx:
                shifts.add(rhs_idx[mod] - lhs_pos)

    best_lhs: Optional[Tuple[float, float]] = None
    best_rhs: Optional[Tuple[float, float]] = None
    best_d = 0.0

    for shift in shifts:
        lr, rr = _find_contiguous_at_shift(lhs_fp, rhs_fp, shift)
        if lr is None or rr is None:
            continue
        d = lr[1] - lr[0]
        if d < min_dur or d > max_dur:
            continue
        if d > best_d:
            best_d = d
            best_lhs = lr
            best_rhs = rr

    if best_lhs is None:
        return None, None

    # Snap intro start to 0 when within 5 s of the beginning (no cold open).
    if best_lhs[0] <= 5.0:
        best_lhs = (0.0, best_lhs[1])
    if best_rhs[0] <= 5.0:
        best_rhs = (0.0, best_rhs[1])

    return best_lhs, best_rhs


def _pairwise_matches(
    fps: Dict[str, np.ndarray],
    paths: List[str],
    min_dur: float,
    max_dur: float,
    max_workers: int,
) -> List[PairMatch]:
    """Compare every episode pair; return every match as ``(a, (a0, a1), b, (b0, b1))``.

    Deterministic: pairs are compared in submission order.
    """
    pairs = [(a, b) for i, a in enumerate(paths) for b in paths[i + 1:]]

    def _compare(pair: Tuple[str, str]):
        a, b = pair
        try:
            return pair, _compare_episodes_chromaprint(
                fps[a], fps[b], min_dur=min_dur, max_dur=max_dur,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Chromaprint compare failed for %s vs %s: %s", a, b, exc)
            return pair, (None, None)

    matches: List[PairMatch] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (a, b), (lr, rr) in pool.map(_compare, pairs):
            if lr is not None and rr is not None:
                matches.append((a, lr, b, rr))
    return matches


# ── Chromaprint orchestrator ────────────────────────────────────────────────


def _detect_by_chromaprint(
    filepaths: List[str],
    durations: Dict[str, float],
    search_window: float = 300.0,
    min_segment: float = 15.0,
    max_segment: float = 120.0,
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """Detect intro/outro boundaries using Chromaprint (intro-skipper algorithm).

    For each pair of episodes the beginning fingerprints are compared to find
    the shared intro and the *reversed* tail fingerprints are compared to find
    the shared credits/outro.
    """
    start_time = time.monotonic()
    stats = FingerprintStats(candidate_files=len(filepaths))

    if len(filepaths) < 2:
        logger.info("Chromaprint requires ≥2 files; got %d. Skipping.", len(filepaths))
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        _set_last_consensus(SeasonConsensus())
        return {}

    intro_fps: Dict[str, np.ndarray] = {}
    outro_fps: Dict[str, np.ndarray] = {}
    valid_paths: List[str] = []

    extraction_jobs: List[Tuple[int, str, float]] = []
    for idx, fp in enumerate(filepaths):
        dur = durations.get(fp)
        if dur is None:
            logger.warning("Duration unknown for %s; skipping in Chromaprint pass", fp)
            continue
        extraction_jobs.append((idx, fp, dur))

    if len(extraction_jobs) < 2:
        logger.warning("Need ≥2 files with known durations; got %d", len(extraction_jobs))
        stats.valid_files = len(extraction_jobs)
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        _set_last_consensus(SeasonConsensus())
        return {}

    def _extract_for_file(job: Tuple[int, str, float]) -> Tuple[int, str, Optional[np.ndarray], Optional[np.ndarray], int, int]:
        idx, fp, dur = job
        intro_len = min(search_window, dur * 0.25, dur)
        intro_fp, intro_cached = _extract_chromaprint(fp, 0.0, intro_len)

        outro_start = max(0.0, dur - search_window)
        outro_len = dur - outro_start
        outro_fp, outro_cached = _extract_chromaprint(fp, outro_start, outro_len)

        hits = int(intro_cached) + int(outro_cached)
        misses = int(not intro_cached) + int(not outro_cached)
        return idx, fp, intro_fp, outro_fp, hits, misses

    max_workers = min(len(extraction_jobs), max(1, (os.cpu_count() or 2) // 2))
    stats.jobs_submitted = len(extraction_jobs)
    stats.workers = max_workers

    extracted: List[Tuple[int, str, Optional[np.ndarray], Optional[np.ndarray], int, int]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_extract_for_file, job): job[1]
            for job in extraction_jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            fp = future_map[future]
            try:
                extracted.append(future.result())
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Chromaprint worker failed for %s: %s", fp, exc)

    # Rebuild in original file order for deterministic behavior.
    extracted.sort(key=lambda row: row[0])

    # ── Extract fingerprints for every episode ──
    for _, fp, intro_fp, outro_fp, hits, misses in extracted:
        stats.cache_hits += hits
        stats.cache_misses += misses

        if intro_fp is not None and outro_fp is not None:
            intro_fps[fp] = intro_fp
            outro_fps[fp] = outro_fp[::-1].copy()  # reverse for credits detection
            valid_paths.append(fp)
        else:
            logger.warning("Chromaprint extraction failed for %s", fp)

    if len(valid_paths) < 2:
        logger.warning("Need ≥2 Chromaprint fingerprints; got %d", len(valid_paths))
        stats.valid_files = len(valid_paths)
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        _set_last_consensus(SeasonConsensus())
        return {}

    stats.valid_files = len(valid_paths)

    # ── Intro/outro detection: season consensus over every pair match ──
    intro = _consensus_segments(
        _pairwise_matches(intro_fps, valid_paths, min_segment, max_segment, max_workers),
        len(valid_paths),
    )
    outro = _consensus_segments(
        _pairwise_matches(outro_fps, valid_paths, min_segment, max(max_segment, search_window), max_workers),
        len(valid_paths),
    )
    def _top_refs(support: Dict[str, int]) -> List[str]:
        return [fp for fp, _ in sorted(support.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]

    consensus = SeasonConsensus(
        intro_len=intro.median_len, outro_len=outro.median_len,
        intro_episodes=len(intro.segments), outro_episodes=len(outro.segments),
        intro_refs=_top_refs(intro.support), outro_refs=_top_refs(outro.support),
    )

    # ── Assemble per-episode results: None where this episode did not match ──
    results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for fp in valid_paths:
        dur = durations[fp]
        is_: Optional[float] = None
        ie: Optional[float] = None
        os_: Optional[float] = None
        if fp in intro.segments:
            is_, ie = intro.segments[fp]
        if fp in outro.segments:
            # Convert reversed-fingerprint coordinates back to absolute time.
            os_ = dur - outro.segments[fp][1]
            if ie is not None and os_ < ie:
                os_ = ie
        results[fp] = (is_, ie, os_)
        consensus.episodes[fp] = MatchInfo(
            intro_support=intro.support.get(fp, 0),
            outro_support=outro.support.get(fp, 0),
            intro_len=(ie - is_) if ie is not None and is_ is not None else None,
            outro_len=(dur - os_) if os_ is not None else None,
            intro_outliers=intro.outliers.get(fp, []),
            outro_outliers=outro.outliers.get(fp, []),
        )
        logger.debug(
            "Chromaprint result for %s: intro=%.1fs–%.1fs, outro=%.1fs",
            Path(fp).name,
            is_ if is_ is not None else 0.0,
            ie if ie is not None else 0.0,
            os_ if os_ is not None else dur,
        )
    _set_last_consensus(consensus)

    logger.info(
        "Chromaprint: %d intro(s), %d outro(s) detected across %d files (season intro %s, credits %s)",
        len(intro.segments), len(outro.segments), len(valid_paths),
        f"{intro.median_len:.1f}s" if intro.median_len is not None else "none",
        f"{outro.median_len:.1f}s" if outro.median_len is not None else "none",
    )

    stats.elapsed_sec = time.monotonic() - start_time
    _set_last_fingerprint_stats(stats)
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _log_band_spectrogram(pcm: np.ndarray) -> Optional[np.ndarray]:
    """40 log-spaced bands × 25 ms frames, each frame z-normalised, so a frame-wise dot product is a cosine similarity."""
    if pcm is None or pcm.size < _XCORR_NFFT:
        return None
    frames = np.lib.stride_tricks.sliding_window_view(pcm, _XCORR_NFFT)[::_XCORR_HOP] * np.hanning(_XCORR_NFFT)
    mag = np.abs(np.fft.rfft(frames, axis=1)).T  # (513, T)
    # ponytail: 44 (not the "obvious" 41) so int-rounding + dedup near the low
    # end still leaves exactly 41 unique edges → 40 bands, matching the
    # documented (40, T) shape.
    edges = np.unique(np.geomspace(4, 512, 44).astype(int))
    bands = np.stack([mag[edges[i]:edges[i + 1]].mean(axis=0) for i in range(len(edges) - 1)])
    spec = np.log1p(bands * 100.0)
    spec -= spec.mean(axis=0, keepdims=True)
    spec /= spec.std(axis=0, keepdims=True) + 1e-6
    return spec


def _xcorr_boundary(ep_path: str, t_ep: float, ref_path: str, t_ref: float, shared: str) -> Optional[float]:
    """Move *t_ep* to where the audio shared with *ref_path* really starts or ends.

    Both files are decoded from ``t - pre`` to ``t + _XCORR_WINDOW`` (same *pre* so the raw
    boundaries line up), the best lag within ±_XCORR_MAX_LAG aligns them, and the boundary is the
    edge of the run of frames whose similarity exceeds _XCORR_SIM_THRESHOLD, walked outward from a
    probe point _XCORR_PROBE s inside the shared side.  *shared* is ``"before"`` when the shared
    audio ends at the boundary (intro_end) and ``"after"`` when it begins there (intro_start,
    outro_start).  Returns ``None`` when audio cannot be decoded or the probe point is not similar.
    """
    fps = _XCORR_SR / _XCORR_HOP
    pre = min(_XCORR_WINDOW, t_ep, t_ref)
    ref_spec = _log_band_spectrogram(_extract_audio_pcm(ref_path, t_ref - pre, pre + _XCORR_WINDOW, sr=_XCORR_SR))
    ep_spec = _log_band_spectrogram(_extract_audio_pcm(ep_path, t_ep - pre, pre + _XCORR_WINDOW, sr=_XCORR_SR))
    if ref_spec is None or ep_spec is None:
        return None
    n = min(ref_spec.shape[1], ep_spec.shape[1])
    ref_spec, ep_spec = ref_spec[:, :n], ep_spec[:, :n]

    max_lag = int(_XCORR_MAX_LAG * fps)
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):  # ep[t] ~ ref[t + lag]
        a0, b0 = max(0, lag), max(0, -lag)
        m = n - abs(lag)
        if m <= 0:
            continue
        score = float((ref_spec[:, a0:a0 + m] * ep_spec[:, b0:b0 + m]).mean())
        if score > best_score:
            best_lag, best_score = lag, score
    a0, b0 = max(0, best_lag), max(0, -best_lag)
    m = n - abs(best_lag)
    sim = (ref_spec[:, a0:a0 + m] * ep_spec[:, b0:b0 + m]).mean(axis=0)
    smooth = np.convolve(sim, np.ones(_XCORR_SMOOTH) / _XCORR_SMOOTH, mode="same")
    good = smooth > _XCORR_SIM_THRESHOLD

    centre = int(round(pre * fps)) - b0  # frame of t_ep in the aligned ep spectrogram
    probe = int(_XCORR_PROBE * fps)
    if shared == "before":
        i = centre - probe
        if i < 0 or i >= m or not good[i]:
            return None
        while i + 1 < m and good[i + 1]:
            i += 1
        edge = i + 1
    else:
        i = centre + probe
        if i < 0 or i >= m or not good[i]:
            return None
        while i - 1 >= 0 and good[i - 1]:
            i -= 1
        edge = i
    return (t_ep - pre) + (edge + b0) * _XCORR_HOP / _XCORR_SR


def _snap_to_silence(
    filepath: str, t: float, duration: float, safe: str = "before",
    before: Optional[float] = None, after: Optional[float] = None,
) -> float:
    """Snap boundary *t* to the start of a nearby silence, preferring the safe side.

    Searches [t − before, t + after] like intro-skipper.  *safe* names the
    side where moving the cut can only keep more content: ``"before"`` for
    intro_end (earlier keeps a little theme), ``"after"`` for intro_start and
    outro_start (later keeps a little cold open or episode).  When *before*/
    *after* are not given they follow *safe*, so the window is biased toward
    the side that is actually searched for a preferred candidate.  Silences on
    the safe side win, closest first; the unsafe side is used only when the
    safe side has none, and then only a silence contiguous with *t* — one
    whose gap to *t* is within a silence's own minimum length, never one past
    spoken words.  Returns *t* unchanged when no silence is found or *t* is at
    the file edges.
    """
    if before is None:
        before = _REFINE_WINDOW_BEFORE if safe == "before" else _REFINE_WINDOW_AFTER
    if after is None:
        after = _REFINE_WINDOW_AFTER if safe == "before" else _REFINE_WINDOW_BEFORE

    if t <= before or t >= duration - after:
        return t

    result = _run_silencedetect(
        filepath,
        _REFINE_NOISE_DB,
        _REFINE_MIN_SILENCE,
        seek=t - before,
        limit=before + after,
    )
    if result is None:
        return t
    starts, ends = result
    if len(ends) < len(starts):
        ends = ends + [float("inf")] * (len(starts) - len(ends))
    pairs = list(zip(starts, ends))
    earlier = [(s, e) for s, e in pairs if s <= t]
    later = [(s, e) for s, e in pairs if s > t]
    preferred, fallback = (earlier, later) if safe == "before" else (later, earlier)
    if preferred:
        return min(preferred, key=lambda p: abs(p[0] - t))[0]

    # Unsafe-side fallback: only a silence contiguous with t (nothing spoken in between).
    if safe == "before":
        candidates = [s for s, e in fallback if s - t <= _REFINE_MIN_SILENCE]
    else:
        candidates = [s for s, e in fallback if e >= t - _REFINE_MIN_SILENCE]
    if candidates:
        return min(candidates, key=lambda start: abs(start - t))
    return t


def _refine_results_with_silence(
    results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
    durations: Dict[str, float],
    cut_margin: float = 0.5,
    references: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """Snap each result's boundaries to nearby silence via :func:`_snap_to_silence`.

    When *references* names reference episodes for a side ("intro"/"outro"),
    each boundary is first cross-correlated against the first usable
    reference (:func:`_xcorr_boundary`) and, if that succeeds, snapped to
    silence within a tight 1 s safe-side-only window; otherwise it falls back
    to a wide silence snap whose window follows the safe side (5 s on the
    safe side, 2 s on the unsafe side).  Runs one file per thread since each ffmpeg call
    only scans a short window.  Falls back to the original (unrefined) tuple
    for a file when snapping would break intro_start < intro_end < outro_start
    ordering.  Then *cut_margin* seconds are given back on every cut
    (intro_start later, intro_end earlier, outro_start later) so an
    off-by-a-little boundary keeps a little theme rather than losing dialogue.
    """
    refinable = [fp for fp in results if durations.get(fp) is not None]
    if not refinable:
        return results

    references = references or {}

    def _boundary(fp: str, t: Optional[float], idx: int, side: str, safe: str, duration: float) -> Optional[float]:
        """Cross-correlate against the first usable reference, then snap on the safe side only.

        Without a usable reference (none configured, or only this episode itself /
        references with no boundary) or when the reference has no shared audio
        there, fall back to a wide silence snap whose window follows the safe
        side (5 s on the safe side, 2 s on the unsafe side).
        """
        if t is None:
            return None
        for ref in references.get(side, []):
            t_ref = results.get(ref, (None, None, None))[idx]
            if ref == fp or t_ref is None:
                continue
            refined = _xcorr_boundary(fp, t, ref, t_ref, "before" if safe == "before" else "after")
            if refined is not None:
                win = (_XCORR_SNAP_WINDOW, 0.0) if safe == "before" else (0.0, _XCORR_SNAP_WINDOW)
                return _snap_to_silence(fp, refined, duration, safe=safe, before=win[0], after=win[1])
            break  # one reference is enough to say the audio is not there
        return _snap_to_silence(fp, t, duration, safe=safe)

    def _refine_one(
        fp: str,
    ) -> Tuple[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
        original = results[fp]
        intro_start, intro_end, outro_start = original
        duration = durations[fp]

        new_intro_start = intro_start
        if intro_start is not None and intro_start > 0.0:
            new_intro_start = _boundary(fp, intro_start, 0, "intro", "after", duration) + cut_margin

        new_intro_end = intro_end
        if intro_end is not None:
            new_intro_end = _boundary(fp, intro_end, 1, "intro", "before", duration) - cut_margin

        new_outro_start = outro_start
        if outro_start is not None:
            new_outro_start = min(duration, _boundary(fp, outro_start, 2, "outro", "after", duration) + cut_margin)

        if (
            new_intro_start is not None
            and new_intro_end is not None
            and new_intro_start >= new_intro_end
        ) or (
            new_intro_end is not None
            and new_outro_start is not None
            and new_outro_start <= new_intro_end
        ):
            logger.debug(
                "Silence-snap broke boundary ordering for %s; keeping original", fp
            )
            return fp, original

        return fp, (new_intro_start, new_intro_end, new_outro_start)

    refined: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = dict(results)
    max_workers = min(len(refinable), max(1, (os.cpu_count() or 2) // 2))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for fp, updated in pool.map(_refine_one, refinable):
            refined[fp] = updated

    return refined


def detect_by_fingerprinting(
    filepaths: List[str],
    durations: Dict[str, float],
    search_window: float = 300.0,
    min_segment: float = 15.0,
    max_segment: float = 120.0,
    cut_margin: float = 0.5,
) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    """Detect intro/outro boundaries for a batch of episode files.

    Compares audio fingerprints from the beginning (intro) and end (outro)
    of each episode against every other episode to find the shared segment.

    Parameters
    ----------
    filepaths:
        Absolute paths to the video files.
    durations:
        Mapping of filepath → duration in seconds (from :func:`get_duration`).
    search_window:
        How many seconds from the start/end of each file to examine.
    min_segment / max_segment:
        Clamp on the detected intro/outro duration.
    cut_margin:
        Seconds given back on every fingerprint cut after silence snapping.

    Returns
    -------
    dict
        Mapping of filepath → ``(intro_start, intro_end, outro_start)``
        in seconds.  Any boundary can be ``None`` when not confidently
        detected.  Files with insufficient audio are omitted.
    """
    # A raised/absent Chromaprint pass must not leak the previous season's consensus.
    _set_last_consensus(SeasonConsensus())

    # ── Chromaprint path (more accurate, inspired by intro-skipper) ──
    if _check_fpcalc():
        try:
            result = _detect_by_chromaprint(
                filepaths, durations, search_window, min_segment, max_segment,
            )
            if result:
                cons = get_last_consensus()
                return _refine_results_with_silence(
                    result, durations, cut_margin=cut_margin,
                    references={"intro": cons.intro_refs, "outro": cons.outro_refs},
                )
            logger.info(
                "Chromaprint detection found no boundaries; "
                "falling back to chroma STFT fingerprinting"
            )
        except Exception as exc:
            logger.warning(
                "Chromaprint detection failed (%s); "
                "falling back to chroma STFT fingerprinting",
                exc,
            )

    # ── Legacy chroma STFT path ──
    start_time = time.monotonic()
    stats = FingerprintStats(candidate_files=len(filepaths))

    if _librosa is None:
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        raise RuntimeError(
            "librosa is required for chroma STFT fingerprinting but is not installed. "
            "Install fpcalc (`brew install chromaprint`) for the primary detection path, "
            "run `pip install 'tvshow-noisemaker[stft]'` for this fallback, "
            "or use --no-fingerprint to disable."
        )

    if len(filepaths) < 2:
        logger.info(
            "Fingerprinting requires ≥2 files; got %d.  Skipping.", len(filepaths)
        )
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        return {}

    sr = _FINGERPRINT_SR
    logger.info(
        "Fingerprinting: computing fingerprints for %d episode(s) in parallel",
        len(filepaths),
    )

    intro_fps_map: Dict[str, np.ndarray] = {}
    outro_fps_map: Dict[str, np.ndarray] = {}
    valid_paths: List[str] = []

    jobs: List[Tuple[str, float]] = []
    for fp in filepaths:
        dur = durations.get(fp)
        if dur is None:
            logger.warning("Duration unknown for %s; skipping in fingerprint pass", fp)
            continue
        jobs.append((fp, dur))

    if len(jobs) < 2:
        logger.warning("Not enough files with known durations for fingerprinting")
        stats.jobs_submitted = len(jobs)
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        return {}

    max_workers = min(len(jobs), max(1, (os.cpu_count() or 2) // 2))
    stats.jobs_submitted = len(jobs)
    stats.workers = max_workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_compute_file_fingerprints, fp, dur, search_window, sr): fp
            for fp, dur in jobs
        }
        for future in concurrent.futures.as_completed(future_map):
            fp = future_map[future]
            try:
                path, intro_fp, outro_fp, hits, misses = future.result()
                stats.cache_hits += hits
                stats.cache_misses += misses
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Fingerprint worker failed for %s: %s", fp, exc)
                continue

            if intro_fp is None or outro_fp is None:
                logger.warning("Could not extract audio from %s; skipping", path)
                continue

            intro_fps_map[path] = intro_fp
            outro_fps_map[path] = outro_fp
            valid_paths.append(path)

    if len(valid_paths) < 2:
        logger.warning("Not enough files with extractable audio for fingerprinting")
        stats.valid_files = len(valid_paths)
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        return {}

    # Use first N valid paths for cross-correlation segment detection
    detection_paths = valid_paths[:_MAX_FINGERPRINT_FILES]
    detection_intro_fps = [intro_fps_map[p] for p in detection_paths]

    intro_bounds = _find_common_segment_bounds(detection_intro_fps, sr, min_segment, max_segment)
    if intro_bounds is None:
        logger.info("Fingerprinting: could not detect a common intro segment")

    # Outro: same logic but fingerprints are reversed so that "end" is at frame 0
    detection_outro_fps_rev = [outro_fps_map[p][:, ::-1] for p in detection_paths]
    outro_bounds = _find_common_segment_bounds(detection_outro_fps_rev, sr, min_segment, max_segment)

    # ── Intro: template-match for precise per-episode boundaries ──
    hop_sec = _HOP_LENGTH / sr
    fp_intro_start_map: Dict[str, float] = {}
    fp_intro_end_map: Dict[str, float] = {}
    if intro_bounds is not None:
        seg_start, seg_len = intro_bounds
        # Extract the detected common segment from the reference fingerprint
        # with proportional padding so that the template captures transition
        # frames at the edges of the theme song.
        ref_fp = detection_intro_fps[0]
        # Padding = 10% of segment length, clamped to [1 s, 10 s]
        pad_sec = max(1.0, min(10.0, seg_len * hop_sec * 0.1))
        pad_frames = int(pad_sec / hop_sec)
        t_start = max(0, seg_start - pad_frames)
        t_end = min(ref_fp.shape[1], seg_start + seg_len + pad_frames)
        template = ref_fp[:, t_start : t_end]
        # Offsets to correct for the padding so that the reported start/end
        # align with the actual theme boundaries, not the padded edges.
        front_pad_sec = (seg_start - t_start) * hop_sec
        back_pad_sec = (t_end - (seg_start + seg_len)) * hop_sec

        if template.shape[1] > 0:
            for path in valid_paths:
                ep_fp = intro_fps_map[path]
                match = _match_template_position(template, ep_fp, sr)
                if match is not None:
                    fp_intro_start_map[path] = match[0] + front_pad_sec
                    fp_intro_end_map[path] = match[1] - back_pad_sec
                    logger.debug(
                        "  Template match %s: intro %.1fs–%.1fs",
                        Path(path).name,
                        fp_intro_start_map[path],
                        fp_intro_end_map[path],
                    )

    # ── Outro: template-match for precise per-episode boundaries ──
    fp_outro_from_end_map: Dict[str, float] = {}
    if outro_bounds is not None:
        out_seg_start, out_seg_len = outro_bounds
        ref_outro_rev = detection_outro_fps_rev[0]
        pad_sec = max(1.0, min(10.0, out_seg_len * hop_sec * 0.1))
        pad_frames = int(pad_sec / hop_sec)
        ot_start = max(0, out_seg_start - pad_frames)
        ot_end = min(ref_outro_rev.shape[1], out_seg_start + out_seg_len + pad_frames)
        outro_template = ref_outro_rev[:, ot_start : ot_end]
        outro_front_pad_sec = (out_seg_start - ot_start) * hop_sec

        if outro_template.shape[1] > 0:
            for path in valid_paths:
                ep_outro_rev = outro_fps_map[path][:, ::-1]
                match = _match_template_position(outro_template, ep_outro_rev, sr)
                if match is not None:
                    from_end = match[0] + outro_front_pad_sec
                    fp_outro_from_end_map[path] = from_end
                    logger.debug(
                        "  Template match outro %s: %.1fs from end",
                        Path(path).name, from_end,
                    )

    # Median values for episodes without individual measurements
    median_intro_start = float(np.median(list(fp_intro_start_map.values()))) if fp_intro_start_map else None
    median_intro_end = float(np.median(list(fp_intro_end_map.values()))) if fp_intro_end_map else None
    median_outro_from_end = (
        float(np.median(list(fp_outro_from_end_map.values())))
        if fp_outro_from_end_map
        else None
    )

    results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for fp in filepaths:
        dur = durations.get(fp)
        if dur is None:
            continue

        is_: Optional[float] = fp_intro_start_map.get(fp)
        if is_ is None and intro_bounds is not None:
            is_ = median_intro_start

        ie: Optional[float] = fp_intro_end_map.get(fp)
        if ie is None and intro_bounds is not None:
            ie = median_intro_end

        ofe = fp_outro_from_end_map.get(fp)
        if ofe is None and outro_bounds is not None:
            ofe = median_outro_from_end

        if ofe is not None:
            outro_start: Optional[float] = dur - ofe
            lower = ie if ie is not None else 0.0
            outro_start = max(lower, min(outro_start, dur))
        else:
            outro_start = None

        results[fp] = (is_, ie, outro_start)

    logger.info(
        "Fingerprinting result – outro_from_end=%s (median); per-episode intro via template matching",
        f"{median_outro_from_end:.1f}s" if median_outro_from_end is not None else "N/A",
    )
    stats.valid_files = len(valid_paths)
    stats.candidate_files = len(filepaths)
    stats.elapsed_sec = time.monotonic() - start_time
    _set_last_fingerprint_stats(stats)
    return _refine_results_with_silence(results, durations, cut_margin=cut_margin)


def _silence_cache_path(
    filepath: str, silence_db: float, min_silence_len: float,
    need_intro: bool = True, need_outro: bool = True,
) -> Optional[Path]:
    """Return a stable cache path for silence detection results.

    Returns ``None`` if the file cannot be stat'd (avoids caching
    nonexistent/mock files).
    """
    try:
        st = os.stat(filepath)
        size = st.st_size
        mtime_ns = st.st_mtime_ns
    except OSError:
        return None
    key = "|".join((
        filepath, f"{silence_db:.1f}", f"{min_silence_len:.3f}",
        f"i{int(need_intro)}o{int(need_outro)}",
        str(size), str(mtime_ns), _CACHE_VERSION,
    ))
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _SILENCE_CACHE_DIR / f"{digest}.json"


def _load_cached_silence(path: Path) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """Load cached silence detection result if available."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (data.get("intro_end"), data.get("outro_start"))
    except Exception:  # pylint: disable=broad-except
        return None


def _save_cached_silence(
    path: Path, intro_end: Optional[float], outro_start: Optional[float],
) -> None:
    """Persist silence detection result to cache (best-effort)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"intro_end": intro_end, "outro_start": outro_start}),
            encoding="utf-8",
        )
    except Exception:  # pylint: disable=broad-except
        return


def _silencedetect_cache_path(
    filepath: str, silence_db: float, min_silence_len: float, seek: float, limit: Optional[float],
) -> Optional[Path]:
    """Cache path for one raw ``silencedetect`` window; ``None`` when *filepath* cannot be stat'd."""
    try:
        st = os.stat(filepath)
    except OSError:
        return None
    key = "|".join((
        filepath, f"{silence_db:.1f}", f"{min_silence_len:.3f}", f"{seek:.3f}",
        "full" if limit is None else f"{limit:.3f}",
        str(st.st_size), str(st.st_mtime_ns), _CACHE_VERSION,
    ))
    digest = hashlib.sha1(key.encode("utf-8"), usedforsecurity=False).hexdigest()
    return _SILENCE_CACHE_DIR / f"raw-{digest}.json"


def _run_silencedetect(
    filepath: str,
    silence_db: float,
    min_silence_len: float,
    seek: float = 0.0,
    limit: Optional[float] = None,
) -> Optional[Tuple[List[float], List[float]]]:
    """Run ffmpeg ``silencedetect`` over one segment of *filepath*.

    Returns absolute-time ``(silence_starts, silence_ends)`` lists, or
    ``None`` if ffmpeg could not be run (so callers can avoid caching a
    transient failure).  With input seeking (``-ss`` before ``-i``) ffmpeg
    reports timestamps relative to the seek point, so *seek* is added back.
    """
    cache_path = _silencedetect_cache_path(filepath, silence_db, min_silence_len, seek, limit)
    if cache_path is not None and cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return list(data["starts"]), list(data["ends"])
        except Exception:  # pylint: disable=broad-except
            pass  # unreadable cache entry: fall through and rerun

    cmd = ["ffmpeg"]
    if seek > 0:
        cmd += ["-ss", str(seek)]
    if limit is not None:
        cmd += ["-t", str(limit)]
    cmd += [
        "-i", filepath,
        "-vn",
        "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_len}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stderr = result.stderr
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("silencedetect failed for %s: %s", filepath, exc)
        return None

    starts: List[float] = []
    ends: List[float] = []
    for line in stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                starts.append(seek + float(line.split("silence_start:")[1].split()[0]))
            except (IndexError, ValueError):
                pass
        elif "silence_end:" in line:
            try:
                ends.append(seek + float(line.split("silence_end:")[1].split("|")[0].strip()))
            except (IndexError, ValueError):
                pass
    if cache_path is not None:
        try:  # ponytail: best-effort cache write, same as _save_cached_silence
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"starts": starts, "ends": ends}), encoding="utf-8")
        except Exception:  # pylint: disable=broad-except
            pass
    return starts, ends


def detect_by_silence(
    filepath: str,
    silence_db: float = -40.0,
    min_silence_len: float = 0.5,
    duration: Optional[float] = None,
    need_intro: bool = True,
    need_outro: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """Detect intro_end and outro_start via ffmpeg ``silencedetect``.

    Only the boundaries requested via *need_intro* / *need_outro* are
    computed; the other comes back ``None``.  When *duration* is known the
    scan is limited to the first ``_SILENCE_INTRO_WINDOW`` seconds and the
    last ``_SILENCE_OUTRO_WINDOW`` seconds instead of decoding the whole
    file.
    """
    if not need_intro and not need_outro:
        return None, None

    cache_path = _silence_cache_path(
        filepath, silence_db, min_silence_len, need_intro, need_outro,
    )
    if cache_path is not None:
        cached = _load_cached_silence(cache_path)
        if cached is not None:
            logger.debug("Silence cache hit for %s", filepath)
            return cached

    intro_starts: List[float] = []
    intro_ends: List[float] = []
    outro_starts: List[float] = []
    failed = False

    if duration is None:
        # Unknown duration: one full-file pass serves both boundaries.
        full = _run_silencedetect(filepath, silence_db, min_silence_len)
        if full is None:
            failed = True
        else:
            intro_starts, intro_ends = full
            outro_starts = intro_starts
    else:
        if need_intro:
            head = _run_silencedetect(
                filepath, silence_db, min_silence_len,
                limit=_SILENCE_INTRO_WINDOW,
            )
            if head is None:
                failed = True
            else:
                intro_starts, intro_ends = head
        if need_outro:
            tail = _run_silencedetect(
                filepath, silence_db, min_silence_len,
                seek=max(0.0, duration - _SILENCE_OUTRO_WINDOW),
            )
            if tail is None:
                failed = True
            else:
                outro_starts = tail[0]

    # Intro end: first significant silence gap in the first few minutes.
    # A gap must end at ≥5 s to be a plausible intro boundary.
    intro_end: Optional[float] = None
    if need_intro:
        for s_start, s_end in zip(intro_starts, intro_ends):
            if s_start > 300.0:       # stop searching past 5 minutes
                break
            if s_end >= 5.0:           # reject trivially short boundaries
                intro_end = s_end
                break

    # Outro start: last detected silence start, rejected if in the first half.
    outro_start: Optional[float] = None
    if need_outro and outro_starts:
        candidate = outro_starts[-1]
        if duration is None or candidate >= duration * 0.5:
            outro_start = candidate

    if cache_path is not None and not failed:
        _save_cached_silence(cache_path, intro_end, outro_start)
    return intro_end, outro_start


def resolve_boundaries(
    filepath: str,
    cfg: "Config",  # type: ignore[name-defined]  # avoid circular import
    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]],
    duration: Optional[float],
) -> Tuple[float, float, float]:
    """Return ``(intro_start, intro_end, outro_start)`` for *filepath*.

    intro_start: where the intro/theme begins (content before this is kept).
    intro_end: where the intro/theme ends (content after this is kept).
    outro_start: where the outro/credits begin (content after this is removed).

    Strategy order:
    1. Per-file timestamps from config (most specific, always wins).
    2. Fingerprinting results (if available for this file).
    3. Silence detection.
    4. User-supplied intro_duration / outro_duration.
    5. Defaults: 0 / 0 / duration (no trimming).
    """
    basename = Path(filepath).name

    # 1. Per-file config timestamps
    ts = cfg.timestamps.get(basename, {})
    ts_intro_start = ts.get("intro_start")
    ts_intro_end = ts.get("intro_end")
    ts_outro = ts.get("outro_start")
    if ts_intro_end is not None and ts_outro is not None:
        logger.debug("Using config timestamps for %s", basename)
        intro_start = float(ts_intro_start) if ts_intro_start is not None else 0.0
        return intro_start, float(ts_intro_end), float(ts_outro)

    # 2. Fingerprinting
    fp_intro_start: Optional[float] = None
    fp_intro_end: Optional[float] = None
    fp_outro: Optional[float] = None
    if filepath in fingerprint_results:
        fp_intro_start, fp_intro_end, fp_outro = fingerprint_results[filepath]

    # 3. Silence detection — only for boundaries that config timestamps and
    #    fingerprinting both failed to provide.
    sil_intro: Optional[float] = None
    sil_outro: Optional[float] = None
    need_sil_intro = ts_intro_end is None and fp_intro_end is None
    need_sil_outro = ts_outro is None and fp_outro is None
    if need_sil_intro or need_sil_outro:
        logger.debug("Running silence detection for %s", basename)
        sil_intro, sil_outro = detect_by_silence(
            filepath,
            silence_db=cfg.silence_db,
            min_silence_len=cfg.min_silence_len,
            duration=duration,
            need_intro=need_sil_intro,
            need_outro=need_sil_outro,
        )

    # Merge: prefer fingerprinting over silence; prefer both over user durations.
    # intro_start is only available from fingerprinting or config timestamps.
    intro_start = _coalesce(
        ts_intro_start,
        fp_intro_start,
        0.0,
    )
    intro_end = _coalesce(
        ts_intro_end,
        fp_intro_end,
        sil_intro,
        cfg.intro_duration,
        0.0,
    )
    fallback_outro = (
        duration - cfg.outro_duration
        if duration is not None and cfg.outro_duration is not None
        else None
    )
    outro_start = _coalesce(
        ts_outro,
        fp_outro if fp_outro is not None and (duration is None or fp_outro < duration) else None,
        sil_outro,
        fallback_outro,
        duration,
    )

    if outro_start is None:
        outro_start = duration or 0.0

    return float(intro_start), float(intro_end), float(outro_start)


def _coalesce(*values: Optional[float]) -> Optional[float]:
    """Return the first non-None value."""
    for v in values:
        if v is not None:
            return v
    return None
