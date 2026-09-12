#!/bin/bash
# SPDX-License-Identifier: BSD-3-Clause

# Script to perform profiling on a command. For example:
#   ./profile_binary.sh ./src/colmap/exe/colmap automatic_reconstructor ...

perf_bin=$(find /usr/lib/linux-tools -name perf | head -1)
if [[ -z $perf_bin ]]; then
    echo "Error: Perf tool not found. Under ubuntu, install as:"
    echo "       sudo apt-get install linux-tools-generic"
    exit 1
fi

"$perf_bin" record -e cycles:u -g "$@"

binary_filename=$(basename -- "${binary_path}")
profile_path="$binary_filename.perf.data"
mv perf.data "$profile_path"

echo "#####################################################################"
echo "###### Profiling finished. Inspect results using the commands: ######"
echo "#####################################################################"
echo "If the perf output contains unknown list items, recompile with RelWithDebInfo"
echo "$perf_bin report -i $profile_path"
echo "$perf_bin report --stdio -g graph,0.5,caller -i $profile_path"
