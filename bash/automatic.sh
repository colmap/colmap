#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /c/path/to/dataset [/c/path/to/dataset/images]" >&2
  exit 1
fi

dataset="$1"
images="${2:-$dataset/images}"
workspace="$dataset/colmap-output"

mkdir -p "$workspace"

colmap automatic_reconstructor \
  --workspace_path "$workspace" \
  --image_path "$images" \
  --data_type individual \
  --quality high \
  --use_gpu 1
