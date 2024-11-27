# Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
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


# This script creates a deployable package of COLMAP for Mac OS.
# It takes the path of the main colmap executable.

set -e
BASE_PATH=$(dirname $1)
APP_PATH="${BASE_PATH}/COLMAP.app"

echo "Creating bundle directory"
mkdir -p "${APP_PATH}/Contents/MacOS"

echo "Copying binary"
cp "$BASE_PATH/colmap" "${APP_PATH}/Contents/MacOS/colmap"

echo "Writing Info.plist"
cat <<EOM >"${APP_PATH}/Contents/Info.plist"
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

# install_name_tool -change @rpath/libtbb.dylib $(brew --prefix tbb)/lib/libtbb.dylib $BASE_PATH/COLMAP.app/Contents/MacOS/COLMAP
# install_name_tool -change @rpath/libtbbmalloc.dylib $(brew --prefix tbb)/lib/libtbbmalloc.dylib $BASE_PATH/COLMAP.app/Contents/MacOS/COLMAP

echo "Linking dynamic libraries"
"$(brew --prefix qt@5)/bin/macdeployqt" "${APP_PATH}"

echo "Wrapping binary"
cat <<EOM >"$APP_PATH/Contents/MacOS/colmap_gui.sh"
#!/bin/bash
script_path="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
if [[ \$(uname -m) == arm64 ]]; then
  for f in \$(ls \${script_path}/../Frameworks/Qt*.framework/Versions/5/Qt*) \$(find \${script_path}/.. -type f -name '*.dylib'); do codesign -s - -f \$f; done
fi
\$script_path/colmap gui
EOM
chmod +x ${APP_PATH}/Contents/MacOS/colmap_gui.sh
sed -i '' 's#<string>colmap</string>#<string>colmap_gui.sh</string>#g' ${APP_PATH}/Contents/Info.plist

echo "Compressing application"
cd "$BASE_PATH"
zip -r "COLMAP.zip" "COLMAP.app"
