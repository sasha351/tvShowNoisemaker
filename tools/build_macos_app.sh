#!/usr/bin/env bash
# Build the self-contained TV Show Noisemaker.app (Python, ffmpeg, ffprobe and fpcalc inside)
# and its drag-to-Applications disk image, dist/TV-Show-Noisemaker-macOS-<arch>.dmg (what people
# install and what the in-app updater downloads), plus dist/TV-Show-Noisemaker-macOS-<arch>.zip:
# apps on 2.2.0 and older only update from that zip.
# Run on a Mac from the repo root; CI runs it on every release tag.
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="$(uname -m)"
NAME="TV Show Noisemaker"
BASE="TV-Show-Noisemaker-macOS-$ARCH"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' tvshow_noisemaker/__init__.py)"
PY="${PYTHON:-python3}"
WORK="build/macos"
TOOLS="$WORK/tools"
FPCALC_VERSION="1.5.1"

rm -rf "$WORK" "dist/$NAME.app" "dist/$NAME" "dist/$BASE".{zip,dmg}
mkdir -p "$TOOLS"

"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet . pyinstaller certifi static-ffmpeg

# ffmpeg + ffprobe: static builds for this CPU, via the static-ffmpeg package's downloader.
"$PY" - "$TOOLS" <<'PY'
import shutil, sys
from static_ffmpeg import run
ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise()
for src in (ffmpeg, ffprobe):
    shutil.copy2(src, sys.argv[1])
PY

# fpcalc (Chromaprint): the upstream universal macOS build.
curl -fsSL "https://github.com/acoustid/chromaprint/releases/download/v$FPCALC_VERSION/chromaprint-fpcalc-$FPCALC_VERSION-macos-universal.tar.gz" \
    | tar -xz -C "$WORK"
cp "$WORK/chromaprint-fpcalc-$FPCALC_VERSION-macos-universal/fpcalc" "$TOOLS/fpcalc"
chmod +x "$TOOLS"/*
xattr -c "$TOOLS"/* 2>/dev/null || true

# Icon
ICONSET="$WORK/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" icons8-convert-144.png --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    sips -z $((size * 2)) $((size * 2)) icons8-convert-144.png --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$WORK/AppIcon.icns"

TVSN_TOOLS_DIR="$PWD/$TOOLS" TVSN_VERSION="$VERSION" TVSN_ICON="$PWD/$WORK/AppIcon.icns" \
    "$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath "$WORK/pyinstaller" \
    packaging/noisemaker.spec

APP="dist/$NAME.app"
# Ad-hoc signature over the whole bundle: Apple Silicon refuses to run unsigned code.
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"

DMG_DIR="$WORK/dmg"
mkdir -p "$DMG_DIR"
cp -R "$APP" "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"
hdiutil create -quiet -volname "$NAME" -srcfolder "$DMG_DIR" -ov -format UDZO "dist/$BASE.dmg"

# Legacy update asset: the 2.2.0 updater downloads this exact name and unpacks it with
# `ditto -x -k`, expecting "$NAME.app" at the top. Drop it once nobody runs 2.2.0.
ditto -c -k --sequesterRsrc --keepParent "$APP" "dist/$BASE.zip"

echo "Built $APP ($VERSION, $ARCH)"
ls -lh "dist/$BASE".{zip,dmg}
