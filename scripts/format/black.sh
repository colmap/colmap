#!/usr/bin/env bash

# This script runs the black Python formatter on the whole repository.

# Check version
version_string=$(black --version | sed -E 's/^.*(\d+\.\d+-.*).*$/\1/')
expected_version_string='21.12'
if [[ "$version_string" =~ "$expected_version_string" ]]; then
    echo "black version '$version_string' matches '$expected_version_string'"
else
    echo "black version '$version_string' doesn't match '$expected_version_string'"
    exit 1
fi

# Get all C++ files checked into the repo, excluding submodules
root_folder=$(git rev-parse --show-toplevel)
all_files=$( \
    git ls-tree --full-tree -r --name-only HEAD . \
    | grep "^.*\(\.py\)$" \
    | sed "s~^~$root_folder/~")
num_files=$(echo $all_files | wc -w)
echo "Formatting ${num_files} files"

# shellcheck disable=SC2086
black --line-length 80 ${all_files}
