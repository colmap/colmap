#!/usr/bin/env bash

# This script applies clang-format to the whole repository.

# Find clang-format
tools='
  clang-format-8
  clang-format
'

clang_format=''
for tool in ${tools}; do
    if type -p "${tool}" > /dev/null; then
        clang_format=$tool
        break
    fi
done

if [ -z "$clang_format" ]; then
    echo "Could not locate clang-format"
    exit 1
fi
echo "Found clang-format: $(which  ${clang_format})"

# Check version
version_string=$($clang_format --version | sed -E 's/^.*([0-9]+\.[0-9]+\.[0-9]+-[0-9]+).*$/\1/')
expected_version_string='8.0.0'
if [[ "$version_string" =~ "$expected_version_string" ]]; then
    echo "clang-format version '$version_string' matches '$expected_version_string'"
else
    echo "clang-format version '$version_string' doesn't match '$expected_version_string'"
    exit 1
fi

# Get all C++ files checked into the repo, excluding submodules
root_folder=$(git rev-parse --show-toplevel)
all_files=$( \
    git ls-tree --full-tree -r --name-only HEAD . \
    | grep "^src.*\(\.cc\|\.h\|\.hpp\|\.cpp\|\.cu\)$" \
    | sed "s~^~$root_folder/~")
num_files=$(echo $all_files | wc -w)
echo "Formatting ${num_files} files"

# Run clang-format
${clang_format} -i $all_files
