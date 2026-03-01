#!/usr/bin/env bash

# This script applies clang-format to the whole repository.

# Check version
version_string=$(clang-format --version | sed -E 's/^.*(\d+\.\d+\.\d+-.*).*$/\1/')
expected_version_string='21.1.8'
if [[ "$version_string" =~ "$expected_version_string" ]]; then
    echo "clang-format version '$version_string' matches '$expected_version_string'"
else
    echo "clang-format version '$version_string' doesn't match '$expected_version_string'"
    exit 1
fi

# Get all C++ files checked into the repo, excluding submodules
root_folder=$(git rev-parse --show-toplevel)
extensions_regex="\(\.cc\|\.h\|\.hpp\|\.cpp\|\.cu\)"
all_files=$( \
    git ls-tree --full-tree -r --name-only HEAD . \
    | grep "\(^src/\(colmap\|glomap\|pycolmap\).*$extensions_regex$\)\|\(benchmark/.*$extensions_regex$\)" \
    | sed "s~^~$root_folder/~")
num_files=$(echo $all_files | wc -w)
echo "Formatting ${num_files} files"

echo "$all_files" | xargs clang-format -i
