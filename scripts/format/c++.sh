#!/usr/bin/env bash

# This script applies clang-format to C++ files in the repository.
# By default, on non-main branches it only formats files changed relative to
# the main branch. On the main branch or with --all, it formats all files.

# Check version
version_string=$(clang-format --version | sed -E 's/^.*(\d+\.\d+\.\d+-.*).*$/\1/')
expected_version_string='22.1.1'
if [[ "$version_string" =~ "$expected_version_string" ]]; then
    echo "clang-format version '$version_string' matches '$expected_version_string'"
else
    echo "clang-format version '$version_string' doesn't match '$expected_version_string'"
    exit 1
fi

root_folder=$(git rev-parse --show-toplevel)
extensions_regex="\(\.cc\|\.h\|\.hpp\|\.cpp\|\.cu\)"
path_regex="\(^src/\(colmap\|glomap\|pycolmap\).*$extensions_regex$\)\|\(benchmark/.*$extensions_regex$\)"

format_all=false
if [[ "$1" == "--all" ]]; then
    format_all=true
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)

if [[ "$format_all" == true ]] || [[ "$current_branch" == "main" ]]; then
    all_files=$( \
        git ls-tree --full-tree -r --name-only HEAD . \
        | grep "$path_regex" \
        | sed "s~^~$root_folder/~")
else
    merge_base=$(git merge-base main HEAD)
    all_files=$( \
        git diff --name-only --diff-filter=d "$merge_base" HEAD \
        | grep "$path_regex" \
        | sed "s~^~$root_folder/~")
fi

if [[ -z "$all_files" ]]; then
    echo "No C++ files to format"
    exit 0
fi
num_files=$(echo "$all_files" | wc -l)
echo "Formatting ${num_files} files"

echo "$all_files" | tr '\n' '\0' | xargs -0 clang-format -i
