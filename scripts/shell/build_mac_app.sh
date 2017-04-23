# COLMAP - Structure-from-Motion and Multi-View Stereo.
# Copyright (C) 2017  Johannes L. Schoenberger <jsch at inf.ethz.ch>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# This script creates a deployable package of COLMAP for Mac OS X.

BIN_PATH="../../install-release/bin"

echo "Creating bundle directory"
mkdir -p "$BIN_PATH/COLMAP.app/Contents/MacOS"

echo "Copying binary"
cp "$BIN_PATH/colmap" "$BIN_PATH/COLMAP.app/Contents/MacOS/COLMAP"

echo "Writing Info.plist"
cat <<EOM >"$BIN_PATH/COLMAP.app/Contents/Info.plist"
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple Computer//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>COLMAP</string>
    <key>CFBundleIdentifier</key>
    <string>COLMAP</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSAppSleepDisabled</key>
    <true/>
</dict>
</plist>
EOM

install_name_tool -change @rpath/libtbb.dylib /usr/local/lib/libtbb.dylib $BIN_PATH/COLMAP.app/Contents/MacOS/COLMAP
install_name_tool -change @rpath/libtbbmalloc.dylib /usr/local/lib/libtbbmalloc.dylib $BIN_PATH/COLMAP.app/Contents/MacOS/COLMAP

echo "Linking dynamic libraries"
/usr/local/opt/qt5/bin/macdeployqt "$BIN_PATH/COLMAP.app"
