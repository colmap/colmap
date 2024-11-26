#!/bin/bash

# Invoke from anywhere to perform an incremental build of pycolmap bindings.
# Make sure to install the requirements from pyproject.toml. If colmap is not
# installed globally but in a custom directory, you should set the colmap_DIR
# environment variable, e.g.:
#
#       colmap_DIR=/path/to/cmake/install/prefix pycolmap/incremental_build.sh

set -e

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

pip install \
    --no-build-isolation \
    -Cbuild-dir="$script_dir/build" \
    -ve \
    "$script_dir"
