#!/usr/bin/env bash
# Build the Windows installer, dist/TV-Show-Noisemaker-Windows-x64-Setup.exe: the app (Python,
# ffmpeg, ffprobe and fpcalc inside) packed with Inno Setup.  Run from the repo root in Git Bash
# on Windows; CI runs it on every release tag.
set -euo pipefail
cd "$(dirname "$0")/.."

NAME="TV Show Noisemaker"
VERSION="$(sed -n 's/^__version__ = "\(.*\)"/\1/p' tvshow_noisemaker/__init__.py)"
PY="${PYTHON:-python}"
WORK="build/windows"
TOOLS="$WORK/tools"
FPCALC_VERSION="1.5.1"

rm -rf "$WORK" "dist/$NAME" dist/TV-Show-Noisemaker-Windows-*
mkdir -p "$TOOLS"

"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet . pyinstaller certifi static-ffmpeg pillow

# ffmpeg + ffprobe: static Windows builds via the static-ffmpeg package's downloader.
"$PY" - "$TOOLS" <<'PY'
import shutil, sys
from static_ffmpeg import run
for src in run.get_or_fetch_platform_executables_else_raise():
    shutil.copy2(src, sys.argv[1])
PY

# fpcalc (Chromaprint): the upstream Windows build.
curl -fsSL -o "$WORK/fpcalc.zip" \
    "https://github.com/acoustid/chromaprint/releases/download/v$FPCALC_VERSION/chromaprint-fpcalc-$FPCALC_VERSION-windows-x86_64.zip"
"$PY" -m zipfile -e "$WORK/fpcalc.zip" "$WORK"
cp "$WORK/chromaprint-fpcalc-$FPCALC_VERSION-windows-x86_64/fpcalc.exe" "$TOOLS/fpcalc.exe"
ls -l "$TOOLS"

# Icon
"$PY" - "$WORK/AppIcon.ico" <<'PY'
import sys
from PIL import Image
Image.open("icons8-convert-144.png").save(sys.argv[1], sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128)])
PY

TVSN_TOOLS_DIR="$PWD/$TOOLS" TVSN_VERSION="$VERSION" TVSN_ICON="$PWD/$WORK/AppIcon.ico" \
    "$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath "$WORK/pyinstaller" \
    packaging/noisemaker.spec

ISCC="/c/Program Files (x86)/Inno Setup 6/ISCC.exe"
if [[ ! -x "$ISCC" ]]; then
    choco install innosetup -y --no-progress >/dev/null
fi
win() { cygpath -w "$PWD/$1"; }
"$ISCC" //Q "//DAppVersion=$VERSION" "//DSourceDir=$(win "dist/$NAME")" "//DOutputDir=$(win dist)" \
    "//DIconFile=$(win "$WORK/AppIcon.ico")" "$(win packaging/windows/installer.iss)"

echo "Built $NAME $VERSION installer"
ls -lh dist/TV-Show-Noisemaker-Windows-x64-Setup.exe
