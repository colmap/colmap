#!/bin/bash
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
