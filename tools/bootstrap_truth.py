#!/usr/bin/env python3
"""Bootstrap tools/ground_truth.json for one show by spectrogram cross-correlation.

Dev tool, not part of the package: needs scipy (present in the venv).

    .venv/bin/python tools/bootstrap_truth.py test_media/Friends03 S03E03

Every episode's first 300 s and last 120 s are cross-correlated against the
reference episode (matched by the substring).  The longest run of similar
frames is the shared audio: its start/end in the episode become intro_start /
intro_end, the start of the shared tail becomes outro_start.  Rules: intro run
>= 15 s and similarity >= 0.75 -> intro cut points (intro_start 0.0 when the
theme starts within 5 s); intro run < 8 s -> no intro (null); shared tail
>= 15 s -> outro_start; anything else is left unchecked (key omitted).  The
reference episode gets the median of its own bounds.  Merges into the truth
file, replacing entries for that show.  Verify by ear before trusting.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tvshow_noisemaker.detector import _extract_audio_pcm  # noqa: E402
from tvshow_noisemaker.pipeline import list_video_files, probe_durations  # noqa: E402

SR, HOP = 8000, 200  # 25 ms frames
HEAD, TAIL = 300.0, 120.0
TRUTH = Path(__file__).resolve().parent / "ground_truth.json"


def spec(path: str, start: float, dur: float) -> np.ndarray:
    """Per-frame z-normalised log-band spectrogram, shape (bands, frames)."""
    pcm = _extract_audio_pcm(path, start, dur, sr=SR)
    _, _, z = stft(pcm, fs=SR, nperseg=1024, noverlap=1024 - HOP, padded=False)
    mag = np.abs(z)
    edges = np.unique(np.geomspace(4, 512, 41).astype(int))
    bands = np.stack([mag[edges[i]:edges[i + 1]].mean(0) for i in range(len(edges) - 1)])
    L = np.log1p(bands * 100)
    L -= L.mean(0, keepdims=True)
    L /= L.std(0, keepdims=True) + 1e-6
    return L


def longest_shared_run(A: np.ndarray, B: np.ndarray, sim_thr: float = 0.45):
    """(run_s, a0_s, a1_s, b0_s, b1_s, mean_sim) of the longest similar stretch at the best lags."""
    _, ta = A.shape
    _, tb = B.shape
    n = 1 << int(np.ceil(np.log2(ta + tb)))
    xc = np.fft.irfft((np.fft.rfft(A, n, axis=1).conj() * np.fft.rfft(B, n, axis=1)).sum(0), n)
    lags = [(i if i < n // 2 else i - n) for i in np.argsort(xc)[::-1]]
    best = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for lag in [lg for lg in lags if -ta < lg < tb][:5]:
        a0, b0 = max(0, -lag), max(0, lag)
        up = min(ta - a0, tb - b0)
        sim = np.convolve((A[:, a0:a0 + up] * B[:, b0:b0 + up]).mean(0), np.ones(8) / 8, mode="same")
        good = sim > sim_thr
        run = (0, 0, 0)
        s = None
        for t in range(up + 1):
            if t < up and good[t]:
                s = t if s is None else s
            elif s is not None:
                if t - s > run[0]:
                    run = (t - s, s, t)
                s = None
        length, s0, s1 = run
        if length * HOP / SR > best[0]:
            best = (length * HOP / SR, (a0 + s0) * HOP / SR, (a0 + s1) * HOP / SR,
                    (b0 + s0) * HOP / SR, (b0 + s1) * HOP / SR, float(sim[s0:s1].mean()))
    return best


def main(show_dir: str, ref_sub: str) -> int:
    files = [f for f in list_video_files(show_dir) if not Path(f).name.startswith(".")]  # AppleDouble junk
    durs = probe_durations(files)
    ref = next(f for f in files if ref_sub in Path(f).name)
    show = Path(show_dir).name
    r_head, r_tail = spec(ref, 0.0, HEAD), spec(ref, durs[ref] - TAIL, TAIL)
    entries, ref_i0, ref_i1, ref_o0 = {}, [], [], []
    for f in files:
        if f == ref:
            continue
        hi = longest_shared_run(r_head, spec(f, 0.0, HEAD))
        ti = longest_shared_run(r_tail, spec(f, durs[f] - TAIL, TAIL))
        e = {}
        if hi[0] >= 15 and hi[5] >= 0.75:
            e["intro_start"] = 0.0 if hi[3] <= 5.0 else round(hi[3], 2)
            e["intro_end"] = round(hi[4], 2)
            ref_i0.append(hi[1])
            ref_i1.append(hi[2])
        elif hi[0] < 8:
            e["intro_start"] = e["intro_end"] = None
        if ti[0] >= 15 and ti[5] >= 0.75:
            e["outro_start"] = round(durs[f] - TAIL + ti[3], 2)
            ref_o0.append(durs[ref] - TAIL + ti[1])
        print(f"{Path(f).name[:44]:<44} intro run {hi[0]:5.1f}s sim {hi[5]:.2f} -> {e.get('intro_start', '?')}-{e.get('intro_end', '?')}"
              f" | tail run {ti[0]:5.1f}s sim {ti[5]:.2f} -> {e.get('outro_start', '?')}", flush=True)
        if e:
            entries[f"{show}/{Path(f).name}"] = e
    ref_entry = {}
    if ref_i0:
        s = statistics.median(ref_i0)
        ref_entry["intro_start"] = 0.0 if s <= 5.0 else round(s, 2)
        ref_entry["intro_end"] = round(statistics.median(ref_i1), 2)
    elif entries and all(v.get("intro_end", 0) is None for v in entries.values()):
        ref_entry["intro_start"] = ref_entry["intro_end"] = None
    if ref_o0:
        ref_entry["outro_start"] = round(min(ref_o0), 2)
    if ref_entry:
        entries[f"{show}/{Path(ref).name}"] = ref_entry

    truth = json.loads(TRUTH.read_text(encoding="utf-8")) if TRUTH.exists() else {}
    truth = {k: v for k, v in truth.items() if k.startswith("_") or not k.startswith(show + "/")}
    truth.update(dict(sorted(entries.items())))
    TRUTH.write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(entries)} entries written for {show} -> {TRUTH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
