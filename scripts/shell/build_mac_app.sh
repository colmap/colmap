#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause

# This script creates a deployable package of COLMAP for macOS.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /path/to/colmap" >&2
    exit 1
fi

BINARY_PATH=$1
BASE_PATH=$(dirname "$BINARY_PATH")
APP_PATH="$BASE_PATH/COLMAP.app"
APP_BINARY="$APP_PATH/Contents/MacOS/colmap"
APP_LAUNCHER="$APP_PATH/Contents/MacOS/colmap_gui.sh"
ARCHIVE_PATH="$BASE_PATH/COLMAP-mac.zip"

rm -rf "$APP_PATH" "$ARCHIVE_PATH"

echo "Creating bundle directory"
mkdir -p "$APP_PATH/Contents/MacOS"

echo "Copying binary"
cp "$BINARY_PATH" "$APP_BINARY"

echo "Writing Info.plist"
cat <<EOM >"$APP_PATH/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>colmap</string>
    <key>CFBundleIdentifier</key>
    <string>COLMAP</string>
    <key>CFBundleName</key>
    <string>COLMAP</string>
    <key>CFBundleDisplayName</key>
    <string>COLMAP</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppSleepDisabled</key>
    <true/>
</dict>
</plist>
EOM

echo "Linking dynamic libraries"
"$(brew --prefix qt)/bin/macdeployqt" "$APP_PATH" -no-codesign

echo "Wrapping binary"
cat <<'EOM' >"$APP_LAUNCHER"
#!/bin/bash
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_path/colmap" gui
EOM
chmod +x "$APP_LAUNCHER"
sed -i '' 's#<string>colmap</string>#<string>colmap_gui.sh</string>#g' "$APP_PATH/Contents/Info.plist"

echo "Signing application binaries"
BREW_PREFIX=$(brew --prefix)
remove_homebrew_rpaths() {
    local binary_path=$1
    while IFS= read -r rpath; do
        if [[ $rpath == "$BREW_PREFIX/"* ]]; then
            install_name_tool -delete_rpath "$rpath" "$binary_path"
        fi
    done < <(otool -l "$binary_path" | awk '/LC_RPATH/{getline; getline; print $2}')
}

while IFS= read -r file_path; do
    if file "$file_path" | grep -q "Mach-O"; then
        remove_homebrew_rpaths "$file_path"
        codesign --force --sign - "$file_path" >/dev/null 2>&1
    fi
done < <(find "$APP_PATH/Contents/Frameworks" "$APP_PATH/Contents/PlugIns" -type f)
remove_homebrew_rpaths "$APP_BINARY"
codesign --force --sign - "$APP_BINARY" >/dev/null 2>&1
codesign --force --sign - "$APP_PATH" >/dev/null 2>&1

echo "Checking packaged binary"
"$APP_BINARY" help >/dev/null
codesign --verify --deep --strict "$APP_PATH"

echo "Compressing application"
ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ARCHIVE_PATH"
