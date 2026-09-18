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
    <string>2.0.0</string>
    <key>CFBundleShortVersionString</key>
    <string>2.0.0</string>
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
cat > "$APP/Contents/MacOS/launcher" <<LAUNCHER
#!/bin/zsh
PYTHONPATH="$SRC" exec "$SRC/.venv/bin/python3" -m tvshow_noisemaker.gui
LAUNCHER
chmod +x "$APP/Contents/MacOS/launcher"

echo "Installed $APP_NAME.app to $APP"
