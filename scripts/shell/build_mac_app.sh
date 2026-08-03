#!/usr/bin/env bash

# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


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
