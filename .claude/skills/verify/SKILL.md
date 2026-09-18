---
name: verify
description: How to run tvshow-noisemaker end-to-end against the bundled test media and read the results
---

# Verify tvshow-noisemaker

Surface is the CLI (the GUI shells out to it). Run against the 21 real
Family Guy episodes in `test_media/FamilyGuy02/`:

```bash
.venv/bin/python -m tvshow_noisemaker.cli \
  -i "test_media/FamilyGuy02" -o <scratch>/out \
  --log-file <scratch>/run.log --verbose
```

Read detection quality from the log, not the console summary:

- `grep "Chromaprint result" run.log` — pre-refinement per-file boundaries.
  Expect 21 lines; Family Guy theme ≈ 30s, so intro spans should cluster
  ~29–34s. `0.0s–0.0s` for E13 (Road to Rhode Island) is correct — it has
  no standard intro. (`is_/ie` print as 0.0 when None.)
- `grep "Fingerprint boundaries" run.log` — post-silence-snap boundaries
  actually used for cutting.
- `grep "Could not determine duration"` — should be empty; failures here
  silently exclude files from fingerprinting (there is a sequential retry).
- Sanity: output mp3 duration ≈ outro_start − intro_end (ffprobe it).

Gotchas:
- Chromaprint/silence results cache under `~/.cache/tvshow_noisemaker/`;
  fingerprint stage is ~2s on cache hits. Silence-snap windows are uncached.
- First-ever read of the mkvs can stall (iCloud materialization); a cold
  run may take minutes in probe/boundaries stages.
- Full run ≈ 1–4 min; encoding 21 mp3s dominates.

## Performance

```bash
tools/bench_perf.sh test_media/FamilyGuy02 [mp3|m4a]
```

Prints a `cold-detect` / `warm-detect` / `warm-encode-<format>` timing matrix.
Cold = `~/.cache/tvshow_noisemaker` wiped first (chromaprint/silence caches
refill on the run); warm = cache hits from a prior run. Detection numbers must
still match `tools/bench_detection.py` — a performance change must never move
them.

## Detection benchmark

```bash
.venv/bin/python tools/bench_detection.py            # all seasons under test_media/
.venv/bin/python tools/bench_detection.py --season "FamilyGuy02"
```

Compares `pipeline.detect` to `tools/ground_truth.json` (seconds; `null` =
boundary does not exist). Read the summary block: "clipped dialogue" must be 0
and "no-intro episodes cut anyway" must be 0 before a detection change ships.
Bootstrap new truth entries with
`tvshow-noisemaker -i <season> -o /tmp/x --detect-only plan.json`, then
verify each cut by ear and copy the corrected numbers in.

Expected per show (bootstrapped truth, cross-correlation refinement, 2026-09-17):
FamilyGuy02 — E13 review ("no intro match"), E05/E20 review ("no credits match"),
E10 review ("no credits match", credits `outro_start` undetected), E11 review
("credits length mismatch"), everything else (16 episodes) high; Friends03 — all
25 high, flag "no shared credits found in this season" (credits are 6–12 s, below
the 15 s match floor); TopGear05 — no intro in the season (intro_end 0.0),
3 episodes with credits high, 4 review ("no credits match").
