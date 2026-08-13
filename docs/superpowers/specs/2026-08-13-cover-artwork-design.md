# Embedded Cover Artwork — Design

Date: 2026-08-13
Status: Approved

## Goal

Converted audio files carry cover art matching the TV show, so they display
artwork when played on an iPod. Artwork is sourced from the Jellyfin images
already present in the media directories, with fallbacks when none exist.

## Behaviour

Artwork is always on. There is no CLI flag and no GUI control.

One cover is chosen per run and embedded into every file that run produces.
This matches the existing metadata scheme, which tags each season as an album
(`artist` = show, `album` = "Season N"), so a season renders as a single tile
in a cover-flow grid.

Artwork resolution never fails a conversion. Every failure path logs a warning
and encodes without art.

WAV output never receives artwork. This is not cosmetic: the WAV muxer rejects
a video stream outright and the encode fails.

```
$ ffmpeg ... -map 0:a:0 -map 1:v -acodec pcm_s16le out.wav
[wav @ ...] wav muxer does not support any stream of type video
[out#0/wav @ ...] Could not write header (incorrect codec parameters ?)
```

## Findings From the Existing Media

Verified against `test_media/Season 02` (22 Family Guy episodes, HEVC MKV) with
ffmpeg 8.1.2.

Every file carries its poster as a Matroska attachment, which ffmpeg exposes as
a video stream tagged with a filename:

```
[STREAM]
index=4
codec_name=mjpeg
codec_type=video
TAG:filename=cover.jpg
TAG:mimetype=image/jpeg
[/STREAM]
```

Three facts drive the design:

1. **The cover stream index varies between files** — index 3 in some episodes,
   index 4 in others. The index must be probed per file, never hardcoded.
2. **Disposition must be requested through `stream_disposition=`, not
   `stream=`.** ffprobe silently omits the field — no error — when it is asked
   for as `-show_entries stream=index,codec_name,disposition`, which makes it
   look as though the cover has no disposition. Asked for correctly, as
   `-show_entries "stream=index,codec_name:stream_disposition=attached_pic:stream_tags=filename,mimetype"`,
   the cover reports `attached_pic: 1` and the primary video stream reports
   `attached_pic: 0`. Both `attached_pic` and the `filename`/`mimetype` tags
   are therefore reliable discriminators on this library.

   (An earlier revision of this spec recorded the opposite, that the cover had
   no disposition. That was an artifact of the wrong field syntax, caught in
   review of Task 3 and verified against `test_media`.)
3. **ffprobe on these files takes several seconds each** (~9s observed). A
   probe per episode would add minutes to a season. Resolving once per run
   keeps it to a single probe.

The embedded covers are 1000x1500 — the same 2:3 shape Jellyfin writes to disk.

## Module: `tvshow_noisemaker/artwork.py`

New module. One public function:

```python
def resolve_artwork(
    input_dir: str,
    sample_video: str,
    intro_end: float,
    workdir: str,
) -> Optional[tuple[str, str]]:
    """Resolve cover art for a run.

    Returns (path_to_normalized_jpeg, source_description) or None.
    """
```

Called once per run. `sample_video` is the first episode queued for encoding;
`intro_end` is that episode's detected intro boundary, already computed in
Phase 1. `workdir` is a caller-owned temporary directory.

The return value's second element is a human-readable source description for
logging, e.g. `"season poster (folder.jpg)"` or `"embedded cover.jpg"`.

### Tier 1 — Jellyfin sidecar image

`input_dir` is usually a season folder (`.../Family Guy/Season 02`), so the
search walks outward from it. First existing file wins:

1. `<input_dir>/{poster,folder,cover,default}.{jpg,jpeg,png}`
2. `<parent>/season<NN>-poster.{jpg,jpeg,png}` — where `NN` is the zero-padded
   season number from `_extract_season_number`, reused from `metadata.py`.
   This is where Jellyfin writes per-season art.
3. `<parent>/{poster,folder,cover,default}.{jpg,jpeg,png}` — the show poster.

Base names are tried in the order listed within each directory. Skipped when
the season number is unknown (step 2 only).

### Tier 2 — Cover embedded in the video

Probe `sample_video` for candidate streams:

```
ffprobe -v error -select_streams v \
  -show_entries "stream=index,codec_name:stream_disposition=attached_pic:stream_tags=filename,mimetype" \
  -of json <sample_video>
```

The single combined `-show_entries` argument is required. Splitting it into
repeated `-show_entries` flags, or asking for `disposition` inside the
`stream=` section, makes ffprobe drop the field silently.

A stream is the cover when `codec_name` is `mjpeg` or `png` **and** it either
has a `filename`/`mimetype` tag or the `attached_pic` disposition set. Both
signals work on this library; keeping the pair means the probe also handles
containers that set only one of them. The lowest matching index is used.

Extract it with the normalization step below, mapping the probed index.

### Tier 3 — Frame grab

Last resort, when the file has neither a sidecar nor an attachment. Seek to
`intro_end + 60` seconds and take one frame from the primary video stream.
Seeking past the intro avoids title cards and the black frames common at the
head of a file.

This tier can still produce a dull or dark frame. No blackframe detection is
included — the tier never fired across the 22 test files, and it exists only so
that something is always embedded. Marked in the source with a `ponytail:`
comment naming the ceiling and the upgrade path (blackframe filter, or a TMDB
lookup) if it turns out to matter.

### Normalization

Every tier produces its output through the same step, so downstream code sees a
uniform image regardless of source. Long edge 600px, aspect preserved, no
cropping:

```
ffmpeg -y -v error <input args> -frames:v 1 \
  -vf "scale=w=600:h=600:force_original_aspect_ratio=decrease" \
  <workdir>/cover.jpg
```

For tier 1 the input args are `-i <sidecar>`; for tier 2, `-i <video> -map 0:<idx>`;
for tier 3, `-ss <intro_end+60> -i <video> -map 0:v:0`.

Measured: a 1000x1500 source becomes 400x600 at ~26 KB.

## Changes to `processor.py`

`process_file()` gains `artwork_path: Optional[str] = None`, threaded to both
command builders alongside the existing `metadata` parameter.

A new `_build_artwork_args()` mirrors the existing `_build_metadata_args()`:

```python
def _build_artwork_args(
    artwork_path: Optional[str], output_format: str
) -> tuple[list[str], list[str]]:
    """Return (input_args, output_args) for embedding cover art."""
```

Returns two empty lists when `artwork_path` is `None` or falsy, or when
`output_format == "wav"`. Otherwise:

- input args: `["-i", artwork_path]`
- output args: `["-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic"]`

The cover input is appended **after** the source `-i` in both builders. This
matters in `_build_simple_cmd`, where `-ss`/`-t` precede the source input and
must continue to apply only to input 0. Because the image is pre-normalized,
the encode copies it rather than filtering it, which also avoids a conflict
between `-filter:v` and the concat path's `-filter_complex`.

`-map 1:v` is correct in both builders: the simple path maps `0:a:0` and the
concat path maps `[out]`, so the image is input 1 either way.

### `-vn` must be removed

Both builders currently pass `-vn`. **`-vn` silently discards the mapped cover
art**, even when `-map 1:v` is explicit, and regardless of whether `-vn`
appears before or after the artwork arguments. Verified both orderings:

```
-map 0:a:0 -vn ... -map 1:v -c:v copy -disposition:v attached_pic  -> audio only
-map 0:a:0 -map 1:v -c:v copy -disposition:v attached_pic -vn ...  -> audio only
```

Without this change the feature produces no artwork at all, with no error.

`-vn` is removed unconditionally from both builders rather than made
conditional on artwork being present. It is redundant: both paths already map
their output streams explicitly (`-map 0:a:0` and `-map [out]`), and both were
verified to produce audio-only output with `-vn` absent and no image input.
Deleting it is simpler than branching on it and leaves one code path to reason
about.

A regression test asserts `-vn` is absent from both builders' output, since
re-adding it would break artwork silently.

### Verified command shapes

Simple path:

```
ffmpeg -y -ss <intro_end> -t <dur> -i <video> -i cover.jpg \
  -map 0:a:0 -acodec libmp3lame -ab 256k -ac 2 \
  -map 1:v -c:v copy -disposition:v attached_pic \
  -id3v2_version 3 -metadata artist=... out.mp3
```

Concat path:

```
ffmpeg -y -i <video> -i cover.jpg \
  -filter_complex "[0:a]atrim=...[pre];[0:a]atrim=...[post];[pre][post]concat=n=2:v=0:a=1[out]" \
  -map "[out]" -acodec libmp3lame -ab 256k -ac 2 \
  -map 1:v -c:v copy -disposition:v attached_pic \
  -id3v2_version 3 -metadata artist=... out.mp3
```

Both were run against real test media and produce a file with an mjpeg stream
alongside the audio. The concat output's duration was confirmed to equal the
sum of its two segments, so adding the image does not disturb the trim.

## Changes to `cli.py`

After Phase 1 detection builds `jobs` and before the encode pool starts:

- Open a `tempfile.TemporaryDirectory()` wrapping the encode phase so the
  normalized JPEG is cleaned up in a `finally`.
- Call `resolve_artwork` with the first job's filepath and intro_end. `jobs`
  entries are plain tuples of
  `(filepath, basename, intro_start, intro_end, outro_start, content_duration, trimmed)`,
  so these are `jobs[0][0]` and `jobs[0][3]`. Skipped when `jobs` is empty.
- Log the outcome at info level: `"Artwork: <source description>"`, or
  `"Artwork: none found — encoding without cover art"`.
- Pass the resolved path into `_encode_one`, which forwards it to
  `process_file`.

The resolved path is read-only and shared across the thread pool, so no
additional locking is needed.

No changes to `gui.py`, `config.py`, or the CLI argument parser.

## Format Support

| Format | Cover art | Notes |
|--------|-----------|-------|
| mp3    | Yes       | Verified; APIC frame via id3v2.3 |
| m4a    | Yes       | Verified; extra `bin_data` stream is the normal iTunes atom |
| flac   | Yes       | Verified |
| wav    | No        | Muxer rejects video streams; encode fails if attempted |

## Error Handling

| Situation | Behaviour |
|-----------|-----------|
| No sidecar, no attachment, frame grab fails | Warn, encode with no art |
| ffprobe fails or returns unparseable JSON | Warn, fall through to next tier |
| Normalization ffmpeg call fails | Warn, fall through to next tier |
| Sidecar exists but is corrupt | Normalization fails, falls through to tier 2 |
| Output format is wav | Silently skip artwork, no warning — by design |

No artwork condition raises. The encode proceeds without a cover in every
failure case.

## Testing

`tests/test_artwork.py`:

- Sidecar precedence: season folder image beats `seasonNN-poster.jpg` beats
  show poster; base-name order within a directory; `.jpg`/`.jpeg`/`.png` all
  recognized. Built from empty files in `tmp_path` with the normalization step
  mocked.
- Season-number-unknown skips the `seasonNN-poster` lookup without error.
- Tier fall-through: with no sidecar present, a mocked ffprobe returning a
  tagged mjpeg stream selects tier 2 and passes the probed index — not a
  hardcoded one — to the extraction call.
- Cover identification rejects the primary video stream and accepts a tagged
  mjpeg stream, including when the tagged stream is at a non-fixed index.
- All tiers failing returns `None` rather than raising.

`tests/test_processor.py` additions:

- `_build_artwork_args` returns empty lists for wav, for `None`, and populated
  lists for mp3/m4a/flac.
- Both command builders place `-i <cover>` after the source input and include
  `-map 1:v`.

Integration: one test encoding a short segment from `test_media` and asserting
via ffprobe that the output contains an mjpeg stream. Skipped when ffmpeg is
unavailable, following the existing suite's convention.
