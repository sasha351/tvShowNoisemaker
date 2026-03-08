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
import os
import subprocess
import time
from dataclasses import asdict, dataclass
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


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def get_duration(filepath: str) -> Optional[float]:
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
            cmd, capture_output=True, text=True, timeout=30
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
            "librosa is required for audio fingerprinting but is not installed. "
            "Run `pip install librosa` or use --no-fingerprint to disable fingerprinting."
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

    n_fft = n_ref + n_other - 1
    corr = np.zeros(n_fft, dtype=np.float64)
    for band in range(ref.shape[0]):
        fa = np.fft.rfft(ref[band].astype(np.float64), n=n_fft)
        fb = np.fft.rfft(other[band].astype(np.float64), n=n_fft)
        corr += np.fft.irfft(fa * np.conj(fb), n=n_fft)

    # Only consider lags within ±max_lag_frames
    mask = np.full(n_fft, -np.inf)
    for k in range(min(max_lag_frames + 1, n_fft)):
        mask[k] = corr[k]
        idx_neg = n_fft - k
        if 0 < idx_neg < n_fft:
            mask[idx_neg] = corr[idx_neg]

    peak = int(np.argmax(mask))
    lag = peak if peak <= n_fft // 2 else peak - n_fft
    return lag


def _find_common_segment_end(
    fingerprints: List[np.ndarray],
    sr: int,
    min_duration: float,
    max_duration: float,
) -> Optional[List[float]]:
    """Find where a common segment ends for each fingerprinted episode.

    Uses cross-correlation to align episodes (handles cold opens where the
    shared theme song starts at a different time in each episode), then
    finds the longest contiguous high-similarity run.

    Returns a list of end-times in seconds (parallel to *fingerprints*),
    or ``None`` if no common segment exceeding *min_duration* is found.
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

    threshold = float(np.percentile(scores[valid] if valid.any() else scores, 60))
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

    # ── Step 5: per-episode end-times ───────────────────────────
    return [(segment_end_ref - lag) * hop_sec for lag in lags]


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
# Public API
# ---------------------------------------------------------------------------


def detect_by_fingerprinting(
    filepaths: List[str],
    durations: Dict[str, float],
    search_window: float = 300.0,
    min_segment: float = 5.0,
    max_segment: float = 180.0,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
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

    Returns
    -------
    dict
        Mapping of filepath → ``(intro_end, outro_start)`` in seconds.
        Either boundary can be ``None`` when not confidently detected.
        Files with insufficient audio are omitted.
    """
    start_time = time.monotonic()
    stats = FingerprintStats(candidate_files=min(len(filepaths), _MAX_FINGERPRINT_FILES))

    if _librosa is None:
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        raise RuntimeError(
            "librosa is required for audio fingerprinting but is not installed. "
            "Run `pip install librosa` or use --no-fingerprint to disable."
        )

    if len(filepaths) < 2:
        logger.info(
            "Fingerprinting requires ≥2 files; got %d.  Skipping.", len(filepaths)
        )
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        return {}

    sr = _FINGERPRINT_SR
    candidate_paths = filepaths[:_MAX_FINGERPRINT_FILES]
    logger.info(
        "Fingerprinting: using first %d episode(s), processing in parallel",
        len(candidate_paths),
    )

    intro_fps: List[np.ndarray] = []
    outro_fps: List[np.ndarray] = []
    valid_paths: List[str] = []

    jobs: List[Tuple[str, float]] = []
    for fp in candidate_paths:
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

            intro_fps.append(intro_fp)
            outro_fps.append(outro_fp)
            valid_paths.append(path)

            if len(valid_paths) >= _MAX_FINGERPRINT_FILES:
                logger.info(
                    "Fingerprinting early exit: reached %d valid file(s)",
                    _MAX_FINGERPRINT_FILES,
                )
                break

    if len(valid_paths) < 2:
        logger.warning("Not enough files with extractable audio for fingerprinting")
        stats.valid_files = len(valid_paths)
        stats.elapsed_sec = time.monotonic() - start_time
        _set_last_fingerprint_stats(stats)
        return {}

    intro_ends = _find_common_segment_end(intro_fps, sr, min_segment, max_segment)
    if intro_ends is None:
        logger.info("Fingerprinting: could not detect a common intro segment")

    # Outro: same logic but fingerprints are reversed so that "end" is at frame 0
    outro_fps_rev = [fp[:, ::-1] for fp in outro_fps]
    outro_from_ends = _find_common_segment_end(outro_fps_rev, sr, min_segment, max_segment)

    # Build per-fingerprinted-episode lookup
    fp_intro_map: Dict[str, float] = {}
    fp_outro_from_end_map: Dict[str, float] = {}
    if intro_ends is not None:
        for path, val in zip(valid_paths, intro_ends):
            fp_intro_map[path] = val
    if outro_from_ends is not None:
        for path, val in zip(valid_paths, outro_from_ends):
            fp_outro_from_end_map[path] = val

    # Median values for episodes that were not fingerprinted
    median_intro = float(np.median(list(fp_intro_map.values()))) if fp_intro_map else None
    median_outro_from_end = (
        float(np.median(list(fp_outro_from_end_map.values())))
        if fp_outro_from_end_map
        else None
    )

    results: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for fp in filepaths:
        dur = durations.get(fp)
        if dur is None:
            continue

        ie: Optional[float] = fp_intro_map.get(fp)
        if ie is None and intro_ends is not None:
            ie = median_intro

        ofe = fp_outro_from_end_map.get(fp)
        if ofe is None and outro_from_ends is not None:
            ofe = median_outro_from_end

        if ofe is not None:
            outro_start: Optional[float] = dur - ofe
            lower = ie if ie is not None else 0.0
            outro_start = max(lower, min(outro_start, dur))
        else:
            outro_start = None

        results[fp] = (ie, outro_start)

    logger.info(
        "Fingerprinting result – intro_end=%s, outro_from_end=%s (per-episode, showing medians)",
        f"{median_intro:.1f}s" if median_intro is not None else "N/A",
        f"{median_outro_from_end:.1f}s" if median_outro_from_end is not None else "N/A",
    )
    stats.valid_files = len(valid_paths)
    stats.elapsed_sec = time.monotonic() - start_time
    _set_last_fingerprint_stats(stats)
    return results


def detect_by_silence(
    filepath: str,
    silence_db: float = -40.0,
    min_silence_len: float = 0.5,
    duration: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Detect intro_end and outro_start via ffmpeg ``silencedetect``.

    Returns ``(intro_end, outro_start)`` in seconds.  Either value may be
    ``None`` if the corresponding boundary could not be determined.
    """
    cmd = [
        "ffmpeg",
        "-i", filepath,
        "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_len}",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        stderr = result.stderr
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("silencedetect failed for %s: %s", filepath, exc)
        return None, None

    silence_starts: List[float] = []
    silence_ends: List[float] = []

    for line in stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                val = float(line.split("silence_start:")[1].split()[0])
                silence_starts.append(val)
            except (IndexError, ValueError):
                pass
        elif "silence_end:" in line:
            try:
                val = float(line.split("silence_end:")[1].split("|")[0].strip())
                silence_ends.append(val)
            except (IndexError, ValueError):
                pass

    # Intro end: look for the first significant silence gap in the first
    # few minutes.  A gap must end at ≥5 s to be a plausible intro boundary
    # (otherwise it's just a brief pause at the very start of the file).
    intro_end: Optional[float] = None
    for s_start, s_end in zip(silence_starts, silence_ends):
        if s_start > 300.0:       # stop searching past 5 minutes
            break
        if s_end >= 5.0:           # reject trivially short boundaries
            intro_end = s_end
            break

    # Outro start: last detected silence start
    outro_start: Optional[float] = None
    if silence_starts:
        candidate = silence_starts[-1]
        if duration is not None and candidate < duration * 0.5:
            # The last silence is in the first half – unlikely to be the outro
            outro_start = None
        else:
            outro_start = candidate

    return intro_end, outro_start


def resolve_boundaries(
    filepath: str,
    cfg: "Config",  # type: ignore[name-defined]  # avoid circular import
    fingerprint_results: Dict[str, Tuple[Optional[float], Optional[float]]],
    duration: Optional[float],
) -> Tuple[float, float]:
    """Return ``(intro_end, outro_start)`` for *filepath* using all strategies.

    Strategy order:
    1. Per-file timestamps from config (most specific, always wins).
    2. Fingerprinting results (if available for this file).
    3. Silence detection.
    4. User-supplied intro_duration / outro_duration.
    5. Defaults: 0 / duration (no trimming).
    """
    basename = Path(filepath).name

    # 1. Per-file config timestamps
    ts = cfg.timestamps.get(basename, {})
    ts_intro = ts.get("intro_end")
    ts_outro = ts.get("outro_start")
    if ts_intro is not None and ts_outro is not None:
        logger.debug("Using config timestamps for %s", basename)
        return float(ts_intro), float(ts_outro)

    # 2. Fingerprinting
    fp_intro: Optional[float] = None
    fp_outro: Optional[float] = None
    if filepath in fingerprint_results:
        fp_intro, fp_outro = fingerprint_results[filepath]

    # 3. Silence detection (fall back when fingerprinting didn't run / failed)
    sil_intro: Optional[float] = None
    sil_outro: Optional[float] = None
    if fp_intro is None or fp_outro is None:
        logger.debug("Running silence detection for %s", basename)
        sil_intro, sil_outro = detect_by_silence(
            filepath,
            silence_db=cfg.silence_db,
            min_silence_len=cfg.min_silence_len,
            duration=duration,
        )

    # Merge: prefer fingerprinting over silence; prefer both over user durations
    intro_end = _coalesce(
        ts_intro,
        fp_intro,
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

    return float(intro_end), float(outro_start)


def _coalesce(*values: Optional[float]) -> Optional[float]:
    """Return the first non-None value."""
    for v in values:
        if v is not None:
            return v
    return None
