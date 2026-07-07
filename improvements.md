# Implementation Plan: Audit Improvements

Handoff document for LLM-driven implementation. Source: full-codebase audit of 2026-07-06
(perf + correctness + simplification). Ten tasks, ranked by impact. Each task block is
**self-contained** — dispatch it to a subagent with the Prelude and nothing else.

## Orchestrator protocol (subagent-driven development)

1. Work tasks in numeric order. T1 must land before T3 (T3 rewrites the same function and
   must preserve T1's `-vn`). T9 must land before T10 (T10's upgrader.sh uses T9's extras).
   All other tasks are independent; if parallelizing, run independent tasks in separate
   worktrees and merge in numeric order.
2. Per task: dispatch a fresh subagent with **only** the Prelude block + that task's block.
   Do not send the whole file. Do not send prior task transcripts.
3. On subagent completion: verify `git log -1 --stat` shows the expected files and run the
   suite gate yourself if the subagent's report is ambiguous. Then dispatch the next task.
4. Line numbers below are hints as of commit `7af9bed` and drift as tasks land — anchor by
   function name, not line number.

## Prelude (include verbatim in every subagent dispatch)

```
Repo: /Users/sasha/Documents/iPod Project/tvShowNoisemaker  (quote the path — it has spaces)
Python CLI (pkg tvshow_noisemaker/): detects TV intro/outro boundaries (Chromaprint fpcalc
primary; librosa chroma-STFT fallback; ffmpeg silencedetect fallback), encodes trimmed MP3s
via ffmpeg subprocesses. Tests mock all subprocess I/O with unittest.mock.

Rules:
- Read ONLY the functions/sections your task names. Do not read whole files or explore.
- Python >=3.9. `from __future__ import annotations` is present in every module.
- Suite gate before commit: cd into the repo and run `.venv/bin/python -m pytest -q`
  → must be 0 failures (~90 tests).
- TDD where the task provides tests: add/replace the tests first, see them fail for the
  stated reason, then apply the edit, then see them pass.
- Commit exactly the files the task lists, with the task's commit message.
- Keep behavior deterministic. Do not refactor beyond the task's scope.
- Report back: files changed, test count, any deviation from the task text.
```

---

## T1 — perf: add `-vn` to silence detection  [no deps]

Why: `detect_by_silence` runs ffmpeg `silencedetect` without `-vn`, so ffmpeg decodes the
**entire video stream** into the null muxer. Biggest single win in the audit.

Read: `tvshow_noisemaker/detector.py` → function `detect_by_silence` (~line 1326);
`tests/test_detector.py` → section "detect_by_silence" (has `_SILENCE_STDERR` fixture).

Test — add to `tests/test_detector.py` after `test_detect_by_silence_caches_results`:

```python
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
```

Expected failure: `AssertionError: missing -vn in: ['ffmpeg', '-i', ...]`

Edit — in `detect_by_silence`, the `cmd` list gains `"-vn"` between `filepath` and `"-af"`:

```python
    cmd = [
        "ffmpeg",
        "-i", filepath,
        "-vn",
        "-af", f"silencedetect=noise={silence_db}dB:d={min_silence_len}",
        "-f", "null", "-",
    ]
```

Gate: `.venv/bin/python -m pytest tests/test_detector.py -q` then full suite.
Commit: `perf: skip video decode in silence detection with -vn`
Files: `tvshow_noisemaker/detector.py tests/test_detector.py`

---

## T2 — perf: encode workers default to full CPU count  [no deps]

Why: encode stage defaults to `cpu_count // 2`; libmp3lame is single-threaded per ffmpeg
process and prior benchmarks (improvements.txt history) showed 6 workers beat 4 by ~28%.
Scope: encode stage only — do NOT touch `boundary_workers` (~cli.py:403).

Read: `tvshow_noisemaker/cli.py` → the encode-stage line
`max_workers = cfg.workers or max(1, (os.cpu_count() or 2) // 2)` (~line 509, in `run`,
just above "Phase 2" comment) and the helper section near `_file_size`;
`tests/test_cli.py` → top import line.

Tests — add to `tests/test_cli.py`; extend the existing import to
`from tvshow_noisemaker.cli import run, _build_parser, _setup_logging, _default_encode_workers`:

```python
def test_default_encode_workers_uses_all_cores():
    with patch("tvshow_noisemaker.cli.os.cpu_count", return_value=8):
        assert _default_encode_workers() == 8


def test_default_encode_workers_unknown_cpu_count():
    with patch("tvshow_noisemaker.cli.os.cpu_count", return_value=None):
        assert _default_encode_workers() == 2
```

Expected failure: `ImportError: cannot import name '_default_encode_workers'`

Edits — add near `_file_size` in `cli.py`:

```python
def _default_encode_workers() -> int:
    """Default worker count for the encode stage.

    libmp3lame is effectively single-threaded per ffmpeg process, so
    audio-only encodes scale close to linearly up to the core count.
    """
    return max(1, os.cpu_count() or 2)
```

and change the encode-stage line to:

```python
    max_workers = cfg.workers or _default_encode_workers()
```

Gate: `.venv/bin/python -m pytest tests/test_cli.py -q` then full suite.
Commit: `perf: default encode workers to full CPU count`
Files: `tvshow_noisemaker/cli.py tests/test_cli.py`

---

## T3 — perf: windowed silence detection + skip already-found boundaries  [after T1]

Why: (a) silence detection scans the whole file but only uses silences in the first ~5 min
and near the end — split into a head pass (`-t 330`) and a tail pass (`-ss dur-600`);
(b) `resolve_boundaries` runs silence detection when *either* fingerprint boundary is
missing and discards the half it didn't need — pass `need_intro`/`need_outro`.
Key detail: with `-ss` before `-i`, silencedetect timestamps are **relative to the seek
point**; the parser adds the seek offset back.

Read: `tvshow_noisemaker/detector.py` → `_silence_cache_path`, `_load_cached_silence`,
`_save_cached_silence`, `detect_by_silence`, `resolve_boundaries`, and the constants block
near `_SILENCE_CACHE_DIR`; `tests/test_detector.py` → sections "detect_by_silence" and
"resolve_boundaries" (uses `_make_cfg` helper).

### Tests

Add helper to `tests/test_detector.py` right after `_SILENCE_STDERR`:

```python
def _fake_silencedetect_run(head_stderr: str, tail_stderr: str):
    """Return a subprocess.run side_effect that distinguishes head vs tail pass."""
    def _side_effect(cmd, **kwargs):
        stderr = tail_stderr if "-ss" in cmd else head_stderr
        return MagicMock(returncode=0, stdout="", stderr=stderr)
    return _side_effect
```

REPLACE `test_detect_by_silence_parses_output`:

```python
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
```

REPLACE `test_detect_by_silence_outro_in_first_half_ignored`:

```python
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
```

REPLACE `test_detect_by_silence_caches_results`:

```python
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
```

ADD three tests:

```python
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
```

Untouched tests `test_detect_by_silence_no_silence` / `test_detect_by_silence_exception`
use `duration=None` (single full-pass branch) and must still pass.

### Edits (detector.py)

1. Constants, next to `_SILENCE_CACHE_DIR`:

```python
# Silence detection scan windows: the intro boundary is only searched in the
# first 5 minutes and the outro near the end, so decode only those regions.
_SILENCE_INTRO_WINDOW = 330.0
_SILENCE_OUTRO_WINDOW = 600.0
```

2. `_silence_cache_path` — add need flags to signature and key (a partial scan must not
satisfy a later full scan). New signature
`def _silence_cache_path(filepath, silence_db, min_silence_len, need_intro=True, need_outro=True) -> Optional[Path]:`
(keep types: `str, float, float, bool, bool`); the `key` tuple gains
`f"i{int(need_intro)}o{int(need_outro)}",` after the `min_silence_len` element. Body
otherwise unchanged.

3. New helper above `detect_by_silence`:

```python
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
    return starts, ends
```

4. REPLACE `detect_by_silence` entirely:

```python
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
```

(The old `_save_cached_silence(...) if cache_path is not None else None` one-liner dies here.)

5. In `resolve_boundaries`, REPLACE the "3. Silence detection" block (the
`if fp_intro_end is None or fp_outro is None:` block) with:

```python
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
```

(`ts_intro_end` / `ts_outro` already exist earlier in the function.)

Gate: `.venv/bin/python -m pytest tests/test_detector.py -q` then full suite.
Commit: `perf: window silence detection and skip boundaries fingerprinting already found`
Files: `tvshow_noisemaker/detector.py tests/test_detector.py`

---

## T4 — fix: dead lists in `_find_common_segment_bounds` + wrong annotations  [no deps]

Why: the function computes per-episode `starts`/`ends` lists no production code reads
(template matching supersedes them), and is annotated as returning a 2-tuple while
returning 4. `detect_by_fingerprinting` is annotated/documented as 2-tuples but returns
3-tuples. Only tests consume the dead lists — rewrite them against the reference-frame
segment.

Read: `tvshow_noisemaker/detector.py` → `_find_common_segment_bounds` (~line 288) and the
two `intro_bounds`/`outro_bounds` unpack sites inside `detect_by_fingerprinting`
(~lines 1178, 1211); `tests/test_detector.py` → the four `test_find_common_segment_bounds_*`
tests.

### Tests (define the new contract; keep each test's setup lines)

`test_find_common_segment_bounds_no_cold_open` — theme at frames 0–100; replace from the
`result =` line down:

```python
    result = _find_common_segment_bounds(fps, sr=11025, min_duration=1.0, max_duration=60.0)
    assert result is not None
    seg_start, seg_len = result
    hop_sec = 512 / 11025
    # Theme spans frames 0–100 in the reference episode.
    assert seg_start * hop_sec < 3.0, f"Expected segment start near 0s, got {seg_start * hop_sec:.1f}s"
    assert abs((seg_start + seg_len) * hop_sec - 100 * hop_sec) < 3.0, (
        f"Expected segment end near {100 * hop_sec:.1f}s, got {(seg_start + seg_len) * hop_sec:.1f}s"
    )
```

`test_find_common_segment_bounds_with_cold_open` — reference fp1 has theme at frames
50–130; replace from the `result =` line down:

```python
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
```

`test_find_common_segment_bounds_short_theme` and `..._long_theme` — change
`_, _, seg_start, seg_len = result` to `seg_start, seg_len = result`.

Expected failure before edits: `ValueError: too many values to unpack (expected 2)`.

### Edits (detector.py)

1. `_find_common_segment_bounds`: return annotation → `Optional[Tuple[int, int]]`; replace
the docstring's Returns paragraph with:

```
    Returns ``(seg_start_frame, seg_len_frames)`` in the coordinate system
    of the *reference* (first) fingerprint, or ``None`` if no common
    segment exceeding *min_duration* is found.  Callers locate per-episode
    boundaries by template-matching this segment against each episode.
```

2. Delete the final "Step 5" block:

```python
    # ── Step 5: per-episode start/end times ─────────────────────
    starts = [max(0.0, (best_start - lag) * hop_sec) for lag in lags]
    ends = [(segment_end_ref - lag) * hop_sec for lag in lags]
    return starts, ends, best_start, best_len
```

→ `    return best_start, best_len`

3. In `detect_by_fingerprinting`:
`intro_starts_list, intro_ends_list, seg_start, seg_len = intro_bounds` →
`seg_start, seg_len = intro_bounds`; and
`_, _, out_seg_start, out_seg_len = outro_bounds` →
`out_seg_start, out_seg_len = outro_bounds`.

4. `detect_by_fingerprinting` return annotation →
`Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]`, and its docstring
Returns section →

```
    Returns
    -------
    dict
        Mapping of filepath → ``(intro_start, intro_end, outro_start)``
        in seconds.  Any boundary can be ``None`` when not confidently
        detected.  Files with insufficient audio are omitted.
```

Gate: full suite.
Commit: `fix: drop dead per-episode lists and correct detection return annotations`
Files: `tvshow_noisemaker/detector.py tests/test_detector.py`

---

## T5 — fix: log file always captures DEBUG  [no deps]

Why: `_setup_logging`'s docstring promises the log file gets DEBUG, but the file handler
is INFO unless `--verbose`, losing the detector's debug diagnostics exactly when needed.

Read: `tvshow_noisemaker/cli.py` → `_setup_logging` (~line 58); `tests/test_cli.py` →
`test_setup_logging_default_console_level_info`.

Test — add to `tests/test_cli.py` after `test_setup_logging_default_console_level_info`
(module already imports `logging`):

```python
def test_setup_logging_file_handler_always_debug(tmp_path):
    """The log file must capture DEBUG even without --verbose."""
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(
        input_dir=str(tmp_path),
        output_dir=str(output_dir),
        log_file=None,
        verbose=False,
    )
    root = logging.getLogger()
    file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
    assert file_handlers, "Expected a FileHandler on the root logger"
    assert all(h.level == logging.DEBUG for h in file_handlers)
```

Expected failure: handler level 20 (INFO) not 10 (DEBUG).

Edit — in `_setup_logging`:

```python
    # File handler – DEBUG only when verbose, otherwise INFO
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
```

→

```python
    # File handler always captures full DEBUG detail for troubleshooting.
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
```

Gate: `.venv/bin/python -m pytest tests/test_cli.py -q` then full suite.
Commit: `fix: log file always captures DEBUG as documented`
Files: `tvshow_noisemaker/cli.py tests/test_cli.py`

---

## T6 — perf: use all Chromaprint pair results; dedupe intro/outro blocks  [no deps]

Why: `_detect_by_chromaprint` submits comparisons of `current` vs *all* remaining episodes,
collects every result, then keeps only the first match and discards the rest — and the
intro/outro blocks are ~40-line copies. One helper keeps the **longest** segment per
episode across all pairs: same CPU, better boundaries, ~70 fewer lines. Behavior change
(longest-across-pairs instead of first-match) is intended and deterministic (`pool.map`
preserves order; ties keep the earlier pair).

Read: `tvshow_noisemaker/detector.py` → `_detect_by_chromaprint` from the comment
`# ── Intro detection: pairwise comparison ──` through the end of the outro
`while remaining:` loop (~lines 880–966), plus `_compare_episodes_chromaprint`'s signature;
`tests/test_detector.py` → Chromaprint helpers section and the detector import block.

Test — add to `tests/test_detector.py`; add `_pairwise_best_segments` to the
`from tvshow_noisemaker.detector import (...)` block:

```python
def test_pairwise_best_segments_keeps_longest():
    """Each episode keeps the longest segment found across all its pairs."""
    fps = {
        "a": np.array([1], dtype=np.uint32),
        "b": np.array([2], dtype=np.uint32),
        "c": np.array([3], dtype=np.uint32),
    }
    names = {1: "a", 2: "b", 3: "c"}
    table = {
        ("a", "b"): ((0.0, 20.0), (0.0, 20.0)),
        ("a", "c"): ((0.0, 30.0), (0.0, 30.0)),
        ("b", "c"): ((0.0, 25.0), (5.0, 28.0)),
    }

    def _fake(lhs, rhs, min_dur, max_dur):
        return table[(names[int(lhs[0])], names[int(rhs[0])])]

    with patch(
        "tvshow_noisemaker.detector._compare_episodes_chromaprint",
        side_effect=_fake,
    ):
        segs = _pairwise_best_segments(fps, ["a", "b", "c"], 15.0, 120.0, max_workers=2)

    assert segs["a"] == (0.0, 30.0)   # longest of 20s (vs b) and 30s (vs c)
    assert segs["b"] == (0.0, 25.0)   # longest of 20s (vs a) and 25s (vs c)
    assert segs["c"] == (0.0, 30.0)   # longest of 30s (vs a) and 23s (vs b)
```

Expected failure: `ImportError: cannot import name '_pairwise_best_segments'`

Edits (detector.py):

1. Add above `_detect_by_chromaprint`:

```python
def _pairwise_best_segments(
    fps: Dict[str, np.ndarray],
    paths: List[str],
    min_dur: float,
    max_dur: float,
    max_workers: int,
) -> Dict[str, Tuple[float, float]]:
    """Compare every episode pair; keep the longest matching segment per episode.

    All pair comparisons were already being computed by the previous
    first-match strategy — this keeps the same work but uses every result.
    Deterministic: pairs are processed in submission order and ties keep
    the earlier pair's segment.
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

    segments: Dict[str, Tuple[float, float]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        for (a, b), (lr, rr) in pool.map(_compare, pairs):
            if lr is None or rr is None:
                continue
            if a not in segments or (lr[1] - lr[0]) > (segments[a][1] - segments[a][0]):
                segments[a] = lr
            if b not in segments or (rr[1] - rr[0]) > (segments[b][1] - segments[b][0]):
                segments[b] = rr
    return segments
```

2. In `_detect_by_chromaprint`, DELETE everything from
`# ── Intro detection: pairwise comparison ──` through the end of the outro
`while remaining:` loop (both `intro_segments`/`outro_segments` blocks, the nested
`_compare_intro_pair`/`_compare_outro_pair` defs, both ThreadPoolExecutor loops).
REPLACE with:

```python
    # ── Intro/outro detection: pairwise comparison, longest match wins ──
    intro_segments = _pairwise_best_segments(
        intro_fps, valid_paths, min_segment, max_segment, max_workers,
    )
    outro_segments = _pairwise_best_segments(
        outro_fps, valid_paths, min_segment, max(max_segment, search_window), max_workers,
    )
```

(`outro_fps` values are already reversed at insertion — downstream conversion unchanged.
Existing test `test_detect_by_chromaprint_parallel_extraction_deterministic_order` must
still pass.)

Gate: full suite.
Commit: `perf: use all chromaprint pair results and deduplicate intro/outro comparison blocks`
Files: `tvshow_noisemaker/detector.py tests/test_detector.py`

---

## T7 — chore: micro-cleanups  [no deps]

Three mechanical edits; existing tests cover them (esp. `test_create_inverted_index`).

1. `tvshow_noisemaker/detector.py` → `_check_fpcalc`:
   `except (FileNotFoundError, Exception):` → `except Exception:  # pylint: disable=broad-except`
2. `tvshow_noisemaker/detector.py` → `_create_inverted_index`: replace body with

```python
def _create_inverted_index(fp: np.ndarray) -> Dict[int, int]:
    """Map each fingerprint point value to the *last* index it appeared at."""
    return dict(zip(fp.tolist(), range(len(fp))))
```

   (`dict` keeps the last value per repeated key = last-index-wins; `tolist()` yields ints.)
3. `tvshow_noisemaker/cli.py` → in `run`'s per-file loop, a second
   `fp_is, fp_ie, fp_os = fingerprint_results.get(filepath, (None, None, None))` appears
   just above `has_fingerprint_boundary = ...` (~line 461). Delete that duplicate line —
   the same unpack already ran at the top of the loop (~line 435); keep
   `has_fingerprint_boundary = fp_ie is not None or fp_os is not None`.

Gate: full suite.
Commit: `chore: simplify exception clause, inverted index, and duplicate lookup`
Files: `tvshow_noisemaker/detector.py tvshow_noisemaker/cli.py`

---

## T8 — chore: `Config` → dataclass  [no deps]

Why: hand-written 10-field `__init__`. Field order must match the current parameter order
(positional construction compatibility). No caller passes `timestamps=None`
(`build_config` defaults `{}`), so the old `timestamps or {}` coercion is not preserved.

Read: `tvshow_noisemaker/config.py` → class `Config` (~lines 41–67) and imports.

Edit — add `from dataclasses import dataclass, field` to imports; replace the class:

```python
@dataclass
class Config:
    """Holds all runtime configuration options."""

    intro_duration: Optional[float] = None
    outro_duration: Optional[float] = None
    # timestamps: mapping of filename (basename) → {"intro_end": float, "outro_start": float}
    timestamps: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mp3_bitrate: int = 192
    silence_db: float = -40.0
    min_silence_len: float = 0.5
    fingerprint_search_window: float = 300.0
    use_fingerprinting: bool = True
    log_file: Optional[str] = None
    workers: Optional[int] = None
```

Gate: `.venv/bin/python -m pytest tests/test_config.py tests/test_detector.py -q` then full suite.
Commit: `chore: convert Config to a dataclass`
Files: `tvshow_noisemaker/config.py`

---

## T9 — chore: librosa → optional `[stft]` extra  [no deps; before T10]

Why: librosa (+ scipy/numba/llvmlite, ~200 MB) exists solely for the chroma-STFT fallback
used when `fpcalc` is missing. The import guard (`try: import librosa` in detector.py)
already exists — only packaging metadata and two error strings change. User approved.

Read: `pyproject.toml`; `tvshow_noisemaker/detector.py` → the two `RuntimeError` messages
mentioning librosa (in `_chroma_fingerprint` and in `detect_by_fingerprinting`);
`README.md` line ~50.

Edits:

1. `pyproject.toml`: remove `"librosa>=0.10.0",` from `dependencies` (leaving numpy +
static-ffmpeg); optional-dependencies becomes:

```toml
[project.optional-dependencies]
stft = [
    "librosa>=0.10.0",
]
dev = [
    "pytest>=7.0",
    "pytest-mock>=3.10",
]
```

2. Both RuntimeError messages become (keep the trailing sentence variant each already has —
`"...to disable fingerprinting."` in `_chroma_fingerprint`, `"...to disable."` in
`detect_by_fingerprinting`):

```python
        raise RuntimeError(
            "librosa is required for chroma STFT fingerprinting but is not installed. "
            "Install fpcalc (`brew install chromaprint`) for the primary detection path, "
            "run `pip install 'tvshow-noisemaker[stft]'` for this fallback, "
            "or use --no-fingerprint to disable fingerprinting."
        )
```

3. `README.md` line ~50:
`    3. Chroma fallback (\`librosa\`) when Chromaprint is unavailable` →
`    3. Chroma fallback (\`librosa\`, install with \`pip install 'tvshow-noisemaker[stft]'\`) when Chromaprint is unavailable`

Gate: `.venv/bin/python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"`
(fallback if no tomllib: `.venv/bin/python -m pip install -e . --dry-run`), then full suite
(dev venv still has librosa; guard path unchanged).
Commit: `chore: move librosa to optional [stft] extra`
Files: `pyproject.toml tvshow_noisemaker/detector.py README.md`

---

## T10 — chore: repo hygiene  [after T9]

Four fixes, one commit. No production code.

1. REPLACE `upgrader.sh` entirely (current bugs: `source source`, `pytest - q`, no shebang,
no error handling):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

git pull --ff-only
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
brew update && brew upgrade ffmpeg chromaprint
pytest -q
```

Verify with `bash -n upgrader.sh && chmod +x upgrader.sh` — do NOT execute it (it pulls,
upgrades brew packages, runs pytest).

2. Untrack `.DS_Store` (currently committed) and ignore it:

```bash
git rm --cached .DS_Store
printf '\n# macOS\n.DS_Store\n' >> .gitignore
```

3. Delete requirements files (duplicate pyproject; T9's extras replace them) and stale
build artifacts (`build/lib` is an outdated package copy that pollutes grep; both dirs
already gitignored):

```bash
git rm requirements.txt requirements-dev.txt
rm -rf build tvshow_noisemaker.egg-info
```

4. Confirm nothing references the requirements files:
`grep -rn "requirements" README.md pyproject.toml tvshow_noisemaker/ tests/ || echo "no references"`
→ expect `no references`.

Gate: full suite.
Commit: `chore: fix upgrader.sh, untrack .DS_Store, drop redundant requirements files`
Files: `upgrader.sh .gitignore` (plus the two `git rm`'d requirements files and cached .DS_Store)

---

## Out of scope (decided)

- Deleting the legacy STFT path entirely — user chose the `[stft]` extra (T9) instead.
- `ProcessPoolExecutor` for `_compare_episodes_chromaprint` (GIL contention) — T6+T7 shrink
  that hot path first; re-benchmark with `--benchmark-fingerprint` before adding it.
- `_fpcalc_segment` temp-WAV double decode — fpcalc can't read a pipe or seek; revisit only
  if profiling shows it matters.
- `pyproject.toml` license/classifiers — needs the user to pick a license.
- Boundary-stage worker count — unbenchmarked; only the encode default changes (T2).
