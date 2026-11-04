# tvShowNoisemaker

Version: 2.4.0

`tvshow-noisemaker` trims TV intros/outros from episode video files and exports tagged MP3s.

## Install

Every release has one installer per platform on the [latest release](https://github.com/sasha351/tvShowNoisemaker/releases/latest) page:

| Platform | File |
|---|---|
| Mac with Apple Silicon (M1 and later) | `TV-Show-Noisemaker-macOS-arm64.dmg` |
| Mac with Intel | `TV-Show-Noisemaker-macOS-x86_64.dmg` |
| Windows 10/11 (64-bit) | `TV-Show-Noisemaker-Windows-x64-Setup.exe` |

GitHub also lists "Source code (zip/tar.gz)" on every release. You don't need those to install the app.

You don't need Python, Homebrew or git on either platform. ffmpeg, ffprobe and fpcalc are inside the app.

### Mac

1. Download the `.dmg` for your Mac:
   - `TV-Show-Noisemaker-macOS-arm64.dmg` for Apple Silicon (M1 and later)
   - `TV-Show-Noisemaker-macOS-x86_64.dmg` for Intel
2. Open the `.dmg` and drag **TV Show Noisemaker** onto **Applications**.
3. The first time only: the app is not notarized by Apple, so macOS blocks it. Open it once, click **Done**, then go to **System Settings → Privacy & Security** and click **Open Anyway** next to the TV Show Noisemaker message. Or run this in Terminal:

   ```bash
   xattr -dr com.apple.quarantine "/Applications/TV Show Noisemaker.app"
   ```

If Homebrew's `ffmpeg` is installed, the app uses it instead, because it has Apple's faster AAC encoder for `m4a`.

To uninstall, drag the app from Applications to the Trash.

### Windows

1. Download `TV-Show-Noisemaker-Windows-x64-Setup.exe` and run it.
2. The installer is not code-signed, so Windows SmartScreen may warn about it. Click **More info**, then **Run anyway**.
3. Follow the installer. It installs for your user only, so it doesn't ask for admin rights. It adds a Start menu entry, and a desktop shortcut if you tick that box.

To uninstall, go to **Settings → Apps → Installed apps → TV Show Noisemaker → Uninstall**.

### Updates

At each launch the app asks GitHub for the newest release. When one exists, an **Update to X.Y.Z** button shows in the footer next to the version. Click it: the app downloads that release's installer for your platform, checks its size and SHA-256, and installs it.

- On Mac, the app copies the new version out of the `.dmg`, quits, replaces itself and reopens.
- On Windows, the app runs the new `Setup.exe` silently. The installer closes the app, upgrades it in place (old files are removed) and reopens it.

It never updates in the middle of a batch. To skip the check, launch with `TVSN_NO_UPDATE=1`.

On Mac, if the app is in a folder your user cannot write to, the update fails with a message in the activity feed. Move the app to `/Applications` (as an admin user) or `~/Applications`.

Mac apps on 2.2.0 look for a `.zip` in each release, and releases now only carry the `.dmg`. Reinstall once from the `.dmg` to get updates again.

## Install from source (CLI, development)

Requirements:

- Python 3.9+ (with Tk for the GUI)
- `ffmpeg` and `ffprobe` (installed automatically through `static-ffmpeg` if missing)
- Recommended: `fpcalc` (Chromaprint)

```bash
git clone https://github.com/sasha351/tvShowNoisemaker.git
cd tvShowNoisemaker
python3 -m venv .venv
.venv/bin/pip install -e .
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

To update a source checkout to the newest release, run `python -m tvshow_noisemaker.updater`. It fast-forwards `main` to the newest `vX.Y.Z` tag and skips the update when that is not safe (another branch, uncommitted changes, local commits).

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

Detection is decided by the season, not by single pairs: the theme is the shared segment most episodes agree on, so a "previously on" recap or a two-parter cannot masquerade as the intro. Episodes that do not share it (a special without the theme, different credits) are flagged for review instead of being cut by guesswork; the CLI skips those with incomplete boundaries and says why, the plan JSON (`--detect-only`) carries the flags. Boundaries are then refined by cross-correlating each episode's audio against a reference episode, so cuts land within about a second. Every cut gives back `--cut-margin` seconds (default 0.5) so a boundary that is slightly off keeps a little theme rather than losing dialogue.

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

# several seasons in one run: S02 is detected while S01 encodes,
# output lands in ./out/Season 01 and ./out/Season 02
tvshow-noisemaker -i "./Show/Season 01" -i "./Show/Season 02" -o ./out

# custom bitrate, workers, and benchmark stats
tvshow-noisemaker -i ./eps -o ./out --mp3-bitrate 320 --workers 8 --benchmark-fingerprint
```

## GUI

For a no-terminal workflow, launch the bundled desktop GUI instead:

```bash
tvshow-noisemaker-gui
```

Add one or more source folders (videos) and pick a destination folder, choose a format and quality, then click Start. Each source folder is processed in turn, writing its output into a subfolder of the destination named after that source folder (to avoid collisions between shows). Detection and encoding run in-process: as soon as one folder's plan is ready, its confident episodes start encoding right away, while the next folder is still being detected.

Episodes the detector isn't confident about (no intro/credits match, a credits-length mismatch, etc.) land in a review panel instead of encoding automatically — tick "Review every episode" before Start to route every episode there regardless. Each flagged row shows the episode name, the reason it was flagged, and three editable time fields (`m:ss.s`, e.g. `1:23.4`) for intro start, intro end and credits start. A ▶ button next to each field plays 6 seconds of audio centered on that time so you can check the cut by ear (■ stops playback). Tick "Skip" to leave an episode out of the export instead of fixing its times. Approve queues that one episode for encoding — editing a time first re-tags the episode `manual`/`high`; "Approve all" does the same for every row still on the panel, stopping at the first one that's still incomplete (blank field, Skip not ticked). "Export approved cuts…" writes everything approved so far to a plan JSON file that the CLI's `--plan` flag accepts, without encoding anything itself.

Click Cancel (the Start button relabels itself once running) to stop the whole batch: in-flight and queued encodes are dropped and the review panel is cleared. For workers, fingerprinting, or config-file control, use the CLI directly.

From a source install, start the GUI with `tvshow-noisemaker-gui`. Built on customtkinter.

### Launcher for a source checkout

If you run from source rather than the downloaded app, `./build_app.sh` installs a `TV Show Noisemaker.app` launcher to `~/Applications` that starts the GUI from your clone. At each launch it fast-forwards the clone to the newest release tag, as described in [Install from source](#install-from-source-cli-development). The footer shows `2.1.0 · 2026-11-04` on a release or `2.1.0 + 3 · abc1234` past one, plus `updates paused` when the update was skipped (the reason is in the activity feed). Rerun the script after you move the repo.

## Releasing

Releases are `vX.Y.Z` tags on `main`. From an up-to-date, clean `main`:

```bash
tools/release.sh 2.2.0
```

The script runs the tests, sets `__version__` (the only place the version lives), commits, tags `v2.2.0` with the commit list as notes, and pushes. The tag starts the **Desktop apps** workflow (`.github/workflows/desktop-apps.yml`). The workflow:

1. runs the tests,
2. builds the Mac app for Apple Silicon and Intel (`tools/build_macos_app.sh`) and the Windows installer (`tools/build_windows_app.sh`, Inno Setup), all with PyInstaller,
3. launches each build once as a smoke test,
4. checks the update paths: on Mac it copies the app out of the `.dmg` and swaps it in place; on Windows it installs, upgrades over the running app, then uninstalls,
5. publishes the GitHub Release with the three installers.

Installed apps find the release on their next launch. Commits pushed to `main` without a tag are never shipped.

The repository must stay public. The app checks for updates and downloads them without credentials, and that does not work on a private repo.

To build locally, run `tools/build_macos_app.sh` on a Mac, or `tools/build_windows_app.sh` in Git Bash on Windows. The output goes to `dist/`.

## CLI Flags (Most Used)

| Flag | Default | Purpose |
|---|---|---|
| `--input-dir`, `-i` | required | Input video directory. Repeat for several seasons; each gets `<output-dir>/<folder name>` |
| `--output-dir`, `-o` | required | Output MP3 directory |
| `--config`, `-c` | none | JSON config file |
| `--format` | mp3 | Output audio format (`mp3`, `m4a`, `flac`, `wav`) |
| `--workers`, `-w` | half CPU count | Parallel encode workers |
| `--mp3-bitrate` | 256 | MP3 bitrate (kbps) |
| `--mp3-quality 0-9` | 5 | LAME quality/speed: 0 slowest/best … 9 fastest. 5 is ~1.2x faster than LAME's default 3, 7 is ~1.6x |
| `--no-stream-copy` | off | m4a: always re-encode, even when the source is already stereo AAC |
| `--fingerprint` / `--no-fingerprint` | on | Enable/disable fingerprinting |
| `--fingerprint-search-window` | 300 | Search window from start/end |
| `--cut-margin SECONDS` | 0.5 | Seconds given back on every detected cut so a slightly-off boundary keeps a little theme instead of losing dialogue |
| `--silence-db` | -40 | Silence threshold (dBFS) |
| `--min-silence-len` | 0.5 | Minimum silence length (seconds) |
| `--benchmark-fingerprint` | off | Print fingerprint timing/cache stats |
| `--verbose`, `-v` | off | Debug terminal logs |
| `--log-file` | auto | Override log file path |
| `--detect-only` | none | Write the per-episode plan (cut points) as JSON and stop; no encoding |
| `--plan` | none | Encode from a plan JSON produced by `--detect-only` (skips detection) |

## Config File

Example:

```json
{
    "mp3_bitrate": 256,
    "mp3_quality": 5,
    "intro_duration": 90.0,
    "outro_duration": 30.0,
    "silence_db": -40.0,
    "min_silence_len": 0.5,
    "fingerprint_search_window": 300.0,
    "cut_margin": 0.5,
    "stream_copy": true,
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

## Output

- One MP3 per input file in `--output-dir`.
- Auto metadata from folder/filename patterns.
- Log file saved to output directory by default.
- `--format m4a` copies the source audio without re-encoding when it is already stereo AAC at or below `--bitrate` (or its bitrate is unknown, as is usual in mkv). That takes seconds per episode instead of a full encode, and adds no generation loss. Cuts land on an AAC frame boundary (about 21 ms). Turn it off with `--no-stream-copy` or `"stream_copy": false`.
- Other m4a sources (5.1, AC-3, higher bitrate) are encoded with Apple's AudioToolbox encoder (`aac_at`) when available, which is much faster than ffmpeg's software AAC.

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
