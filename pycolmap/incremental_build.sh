#!/bin/bash

set -e

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

pip install \
    --no-build-isolation \
    -Cbuild-dir="$script_dir/build" \
    -ve \
    "$script_dir"
