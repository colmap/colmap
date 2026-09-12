#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause

# Script to generate a coverage report for the COLMAP code base. Must be
# executed from the build directory. The generated HTML report will be available
# in the coverage/ directory.

# The script assumes that the codebase has been compiled with CMake options:
#   cmake ...
#       -DTESTS_ENABLED=ON \
#       -DCOVERAGE_ENABLED=ON \
#       -DCMAKE_BUILD_TYPE=RelWithDebInfo|Debug
#   ninja
#   ctest -j$(nproc)

colmap_root_dir=$(git rev-parse --show-toplevel)

if [ ! -f "CMakeCache.txt" ]; then
    echo "Please run this script from the build directory."
    exit 1
fi

rm -rf coverage-html
mkdir -p coverage-html
gcovr \
    --root "$colmap_root_dir" \
    --exclude "$colmap_root_dir/src/thirdparty/*" \
    --exclude "$(pwd)/_deps/*" \
    --exclude "$(pwd)/src/colmap/ui/colmap_ui_autogen/*" \
    --cobertura coverage-cobertura.xml \
    --cobertura-pretty \
    --html-nested coverage-html/index.html \
    --html-theme github.blue
