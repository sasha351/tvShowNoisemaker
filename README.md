# tvShowNoisemaker

A Python CLI tool that processes a folder of TV episode video files and produces
trimmed `.mp3` files with intros and outros removed.

---

## Features

- **Batch processing** – point it at a folder; every video file is processed automatically.
- **Wide format support** – accepts `.mp4`, `.mkv`, `.avi`, `.mov`, `.wmv`, `.flv`, `.webm`, `.m4v`, `.ts`, `.mpg`, `.mpeg`, `.3gp`, `.ogv`, `.vob`.
- **Three detection strategies** (in priority order):
    1. **Audio fingerprinting** (librosa chroma features) – identifies identical/near-identical intro and outro music from the first 5 episodes. Fingerprinting runs in parallel and caches fingerprints on disk to speed up repeat runs.
  2. **Silence detection** (ffmpeg `silencedetect`) – falls back to silence boundaries when fingerprinting fails or only one file is provided.
  3. **User-supplied timestamps/durations** – a per-episode timestamp table in the config file, or global `--intro-duration` / `--outro-duration` flags.
- **Configurable MP3 bitrate** (default 192 kbps).
- **Skip + log** – failed or invalid files are skipped and logged; the batch never crashes.
- **Safe detection fallback** – if no intro/outro boundary can be detected, that file is skipped instead of converting the full episode.
- **Optional JSON config file** for reproducible runs.

---

## Requirements

- Python ≥ 3.9
- **ffmpeg & ffprobe** – required for all audio processing
- Python packages (installed automatically): `librosa`, `numpy`, `static-ffmpeg`

### ffmpeg

**Automatic (recommended):** `ffmpeg` is bundled via the
[`static-ffmpeg`](https://pypi.org/project/static-ffmpeg/) Python package,
which is installed automatically with `pip install`. No extra steps needed –
the tool will download a platform-appropriate ffmpeg binary on first run.

**Manual (alternative):** If you prefer a system-wide install:

| Platform | Command |
|---|---|
| macOS (Homebrew) | `brew install ffmpeg` |
| Ubuntu / Debian | `sudo apt install ffmpeg` |
| Fedora | `sudo dnf install ffmpeg` |
| Windows (Chocolatey) | `choco install ffmpeg` |
| Windows (Scoop) | `scoop install ffmpeg` |

When both a system `ffmpeg` and the bundled one are available, the system
installation takes priority.

---

## Installation

```bash
pip install -e .
```

Or install from `requirements.txt` without editable mode:

```bash
pip install -r requirements.txt
pip install .
```

This will install all dependencies including `static-ffmpeg`, which provides
`ffmpeg` and `ffprobe` automatically.

---

## Quick Start

```bash
# Basic usage – auto-detect intros/outros via fingerprinting
tvshow-noisemaker --input-dir /path/to/episodes --output-dir /path/to/mp3s

# Disable fingerprinting, rely on silence detection only
tvshow-noisemaker --input-dir ./eps --output-dir ./out --no-fingerprint

# Supply fallback durations when detection is unreliable
tvshow-noisemaker --input-dir ./eps --output-dir ./out \
    --intro-duration 90 --outro-duration 30

# Use a config file
tvshow-noisemaker --input-dir ./eps --output-dir ./out --config config.json

# Custom bitrate + save log
tvshow-noisemaker --input-dir ./eps --output-dir ./out \
    --mp3-bitrate 320 --log-file ./run.log
```

---

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--input-dir`, `-i` | *(required)* | Folder containing video files (mp4, mkv, avi, mov, etc.) |
| `--output-dir`, `-o` | *(required)* | Destination folder for `.mp3` files (created if absent) |
| `--intro-duration SECS` | `None` | Fallback intro duration in seconds |
| `--outro-duration SECS` | `None` | Fallback outro duration in seconds from end |
| `--mp3-bitrate KBPS` | `192` | MP3 encoding bitrate |
| `--config FILE` | `None` | Path to JSON config file |
| `--log-file FILE` | `None` | Override log file path (default log file is auto-created in output dir) |
| `--fingerprint` | *(on)* | Use audio fingerprinting (default) |
| `--no-fingerprint` | – | Disable fingerprinting |
| `--silence-db DB` | `-40.0` | Silence threshold in dBFS |
| `--min-silence-len SECS` | `0.5` | Minimum silence duration |
| `--fingerprint-search-window SECS` | `300` | Seconds from start/end to examine |
| `--verbose`, `-v` | – | Enable debug logging |

Terminal logging defaults to `INFO+`; `--verbose` raises terminal output to `DEBUG`.
The log file always captures `DEBUG` details.

---

## Config File Format

```json
{
    "mp3_bitrate": 192,
    "intro_duration": 90.0,
    "outro_duration": 30.0,
    "silence_db": -40.0,
    "min_silence_len": 0.5,
    "fingerprint_search_window": 300.0,
    "timestamps": {
        "s01e01.mp4": { "intro_end": 88.5, "outro_start": 2610.0 },
        "s01e02.mp4": { "intro_end": 90.0, "outro_start": 2605.5 }
    }
}
```

| Key | Type | Description |
|---|---|---|
| `mp3_bitrate` | int | MP3 bitrate in kbps |
| `intro_duration` | float | Global fallback intro duration (seconds) |
| `outro_duration` | float | Global fallback outro duration from end (seconds) |
| `silence_db` | float | Silence threshold in dBFS |
| `min_silence_len` | float | Minimum silence gap (seconds) |
| `fingerprint_search_window` | float | Seconds to examine from start/end |
| `timestamps` | object | Per-file overrides (filename → `intro_end`/`outro_start`) |

Per-file `timestamps` entries always win over all other detection methods.
CLI flags override config-file values.

---

## Detection Method Tradeoffs

| Method | Accuracy | Speed | Requirements |
|---|---|---|---|
| Audio fingerprinting | ★★★★★ | ★★☆☆☆ (CPU-bound) | ≥ 2 episodes, librosa |
| Silence detection | ★★★☆☆ | ★★★★★ | ffmpeg |
| User timestamps | ★★★★★ | ★★★★★ | Manual inspection |

**Recommendation**: Use the default fingerprinting for a full season folder.
If you only have a single episode, provide `--intro-duration` and
`--outro-duration` or a `--config` timestamps file.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## Output

- One `.mp3` per input video file, same filename, written to `--output-dir`.
- A log file is always created (auto-named in `--output-dir` unless `--log-file` is set) listing detailed processing, skipped files, and failures.
