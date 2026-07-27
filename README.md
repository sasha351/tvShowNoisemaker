# tvShowNoisemaker

Version: 1.1.0

`tvshow-noisemaker` trims TV intros/outros from episode video files and exports tagged MP3s.

## Installation

Requirements:

- Python 3.9+
- `ffmpeg` and `ffprobe`
- Recommended: `fpcalc` (Chromaprint)

Install project dependencies:

```bash
pip install -e .
```

Install optional system packages:

```bash
brew install ffmpeg chromaprint
```

Create a reliable launcher (recommended):

```bash
mkdir -p "$HOME/bin"
SRC="/path/to/tvShowNoisemaker"   # replace with your clone location
cat > "$HOME/bin/tvshow-noisemaker" <<EOF
#!/bin/zsh
VENV="$SRC/.venv"
PYTHONPATH="$SRC" exec "\$VENV/bin/python3" -m tvshow_noisemaker.cli "\$@"
EOF
chmod +x "$HOME/bin/tvshow-noisemaker"
```

Make sure `$HOME/bin` is in your `PATH`.

If conda is installed, the pip-generated entry point (`.venv/bin/tvshow-noisemaker`) can intermittently fail with `ModuleNotFoundError` outside the project directory. This is a known conda + editable install interaction where `.pth` handling can be inconsistent. The wrapper above is the reliable fix because it runs `python -m` with `PYTHONPATH` set explicitly.

## What It Does

- Batch processes a folder of episodes.
- Detects intro/outro boundaries using this priority:
    1. Per-file timestamps from config
    2. Chromaprint fingerprinting (`fpcalc`)
    3. Chroma fallback (`librosa`, install with `pip install 'tvshow-noisemaker[stft]'`) when Chromaprint is unavailable
    4. Silence detection (`ffmpeg silencedetect`)
    5. User fallback durations
- Encodes to MP3 with metadata (artist/album/title/track/genre).
- Writes a per-run log file with summary and stage timings.

## Usage

Run a full command with both input and output directories:

```bash
tvshow-noisemaker \
    -i "$HOME/Media/TV Shows/Show Name/Season 01" \
    -o "$HOME/Media/TV Show Clips/Show Name/Season 01"
```

If writing to an external volume, ensure the base output directory already exists before running:

```bash
mkdir -p "/Volumes/MY DRIVE/Media/TV Show Clips"
tvshow-noisemaker \
    -i "$HOME/Media/TV Shows/Show Name/Season 01" \
    -o "/Volumes/MY DRIVE/Media/TV Show Clips/Show Name/Season 01"
```

Paths containing apostrophes must be quoted carefully:

```bash
# Correct - double quotes around path containing apostrophe
-o "/Volumes/MY DRIVE'S VOLUME/Media/TV Show Clips/Show Name/Season 01"

# Wrong - apostrophe breaks shell parsing
-o /Volumes/MY\ DRIVE\'S\ VOLUME/Media/...
```

Common variants:

```bash
# no fingerprinting
tvshow-noisemaker -i ./eps -o ./out --no-fingerprint

# fallback durations
tvshow-noisemaker -i ./eps -o ./out --intro-duration 90 --outro-duration 30

# custom bitrate, workers, and benchmark stats
tvshow-noisemaker -i ./eps -o ./out --mp3-bitrate 320 --workers 8 --benchmark-fingerprint
```

## GUI

For a no-terminal workflow, launch the bundled desktop GUI instead:

```bash
tvshow-noisemaker-gui
```

Add one or more source folders (videos) and pick a destination folder (MP3s), then click Start. Each source folder is processed in turn, writing its MP3s into a subfolder of the destination named after that source folder (to avoid collisions between shows). The GUI runs the same CLI underneath and streams its log output live, with a progress bar and status line showing which folder is currently processing. Click Cancel to stop the whole batch early. There are no options beyond source/destination — for bitrate, workers, fingerprinting, or config-file control, use the CLI directly.

Built entirely on tkinter (Python stdlib) — no extra install required.

### Desktop App Shortcut

For a double-clickable launcher (no terminal at all), run once:

```bash
./build_app.sh
```

This installs `TV Show Noisemaker.app` to `~/Applications`, visible in Spotlight/Launchpad. Drag it to the Dock for a permanent icon. Rerun the script any time (e.g. after moving the repo) to rebuild it.

## CLI Flags (Most Used)

| Flag | Default | Purpose |
|---|---|---|
| `--input-dir`, `-i` | required | Input video directory |
| `--output-dir`, `-o` | required | Output MP3 directory |
| `--config`, `-c` | none | JSON config file |
| `--workers`, `-w` | CPU count | Parallel encode workers |
| `--ffmpeg-threads` | CPU count ÷ workers | Threads per ffmpeg process (`0` = ffmpeg auto) |
| `--mp3-bitrate` | 256 | MP3 bitrate (kbps) |
| `--fingerprint` / `--no-fingerprint` | on | Enable/disable fingerprinting |
| `--fingerprint-search-window` | 300 | Search window from start/end |
| `--silence-db` | -40 | Silence threshold (dBFS) |
| `--min-silence-len` | 0.5 | Minimum silence length (seconds) |
| `--benchmark-fingerprint` | off | Print fingerprint timing/cache stats |
| `--verbose`, `-v` | off | Debug terminal logs |
| `--log-file` | auto | Override log file path |

## Config File

Example:

```json
{
    "mp3_bitrate": 256,
    "intro_duration": 90.0,
    "outro_duration": 30.0,
    "silence_db": -40.0,
    "min_silence_len": 0.5,
    "fingerprint_search_window": 300.0,
    "ffmpeg_threads": 1,
    "timestamps": {
        "s01e01.mp4": { "intro_end": 88.5, "outro_start": 2610.0 }
    }
}
```

Notes:

- `timestamps` entries are highest priority.
- CLI flags override config values.

## Performance and Benchmarking

Use benchmark output while tuning workers:

```bash
tvshow-noisemaker -i ./eps -o ./out --workers 8 --benchmark-fingerprint
```

The run summary includes:

- fingerprint cache hits/misses
- stage timings (`probe`, `fingerprint`, `boundaries`, `encode`)
- throughput and total runtime

### Thread policy

Warm-cache runs are encode-bound. Left alone, ffmpeg gives each process about one
thread per core, so `--workers N` asks the machine for roughly `N × cores` threads and
spends the surplus on scheduler contention rather than work. By default the encode pool
therefore divides the cores across its workers (`--ffmpeg-threads` = CPU count ÷
workers, minimum 1), keeping the pool near one thread per core overall.

Both settings are hard overrides, so you can A/B them:

```bash
# restore the old behaviour (ffmpeg decides per process)
tvshow-noisemaker -i ./eps -o ./out --workers 8 --ffmpeg-threads 0

# compare CPU efficiency between policies
/usr/bin/time -l tvshow-noisemaker -i ./eps -o ./out --workers 8
```

Compare `(user + sys) / real` across runs — lower means less time lost to contention at
the same wall time. Thread count does not affect output bytes.

## Output

- One MP3 per input file in `--output-dir`.
- Auto metadata from folder/filename patterns.
- Log file saved to output directory by default.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Troubleshooting

### ModuleNotFoundError when launching command

If `tvshow-noisemaker` works only from the project directory (or fails intermittently outside it), use the `$HOME/bin/tvshow-noisemaker` wrapper from the installation section. This avoids the pip-generated entry point behavior that can be affected by conda when editable installs are used.

### PermissionError creating output directory

The tool creates the output path with `mkdir(parents=True)`, but macOS volume permissions can still block creating top-level folders on external volume roots. Create the base directory first, then output into a subdirectory beneath it.

### Paths with apostrophes or special characters fail

Wrap the full path in double quotes when it contains apostrophes or spaces (for example, `"/Volumes/MY DRIVE'S VOLUME/..."`). Unquoted paths can break shell parsing.
