#!/bin/zsh
# Builds a double-clickable macOS app for the GUI and installs it to ~/Applications.
# Safe to rerun (e.g. after moving the repo or updating the icon).
set -euo pipefail

SRC="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="TV Show Noisemaker"
APP="$HOME/Applications/$APP_NAME.app"
ICON_SOURCE="$SRC/icons8-convert-144.png"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- Icon ---
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ICON_SOURCE" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" "$ICON_SOURCE" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

# --- Info.plist ---
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.tvshownoisemaker.gui</string>
    <key>CFBundleVersion</key>
    <string>1.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>launcher</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
PLIST

# --- Launcher ---
# Record the interpreter used at build time so the app can fall back to it when
# no project .venv exists (e.g. conda / system-Python installs).
if [[ -x "$SRC/.venv/bin/python3" ]]; then
    BUILD_PYTHON="$SRC/.venv/bin/python3"
else
    BUILD_PYTHON="$(command -v python3 || true)"
fi

# Config lines are written with expansion; the rest of the script is a quoted
# heredoc so its own $variables survive to run time.
{
    echo '#!/bin/zsh'
    echo "SRC=\"$SRC\""
    echo "BUILD_PYTHON=\"$BUILD_PYTHON\""
    cat <<'LAUNCHER'

# LaunchServices (Spotlight / Launchpad / Dock / Finder) starts apps with a
# minimal PATH — add the usual spots so python3 and its tools are findable.
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/bin:$PATH"

fail() {
    /usr/bin/osascript -e "display alert \"TV Show Noisemaker\" message \"$1\"" >/dev/null 2>&1
    exit 1
}

# Pick the first Python 3 that actually exists: the project venv, the
# interpreter this app was built with, then whatever is on PATH.
PYBIN=""
for candidate in "$SRC/.venv/bin/python3" "$BUILD_PYTHON" "$(command -v python3)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        PYBIN="$candidate"
        break
    fi
done

[[ -n "$PYBIN" ]] || fail "No Python 3 interpreter found. Install Python 3, or create a .venv in the project folder and re-run build_app.sh."

# Preflight: make sure the chosen interpreter can import the GUI's deps, so a
# missing package shows an actionable dialog instead of a silent non-launch.
if ! ERR="$("$PYBIN" -c 'import customtkinter' 2>&1)"; then
    fail "The Python at $PYBIN is missing required packages:

$ERR

Install them with:  \"$PYBIN\" -m pip install -e \"$SRC\""
fi

PYTHONPATH="$SRC" exec "$PYBIN" -m tvshow_noisemaker.gui
LAUNCHER
} > "$APP/Contents/MacOS/launcher"
chmod +x "$APP/Contents/MacOS/launcher"

# Register with LaunchServices so the app shows up in Spotlight/Launchpad
# immediately instead of after the next reindex.
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
if [[ -x "$LSREGISTER" ]]; then
    "$LSREGISTER" -f "$APP" || true
fi

echo "Installed $APP_NAME.app to $APP"
