#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /c/path/to/dataset [/c/path/to/dataset/images]" >&2
  exit 1
fi

dataset="$1"
images="${2:-$dataset/images}"
db="$dataset/database.db"
sparse="$dataset/sparse"
dense="$dataset/dense"

mkdir -p "$sparse" "$dense"

colmap feature_extractor \
  --database_path "$db" \
  --image_path "$images" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.max_num_features 12000 \
  --SiftExtraction.domain_size_pooling 1 \
  --SiftExtraction.estimate_affine_shape 1

colmap exhaustive_matcher \
  --database_path "$db" \
  --FeatureMatching.guided_matching 1 \
  --SiftMatching.max_ratio 0.85

colmap mapper \
  --database_path "$db" \
  --image_path "$images" \
  --output_path "$sparse"

colmap image_undistorter \
  --image_path "$images" \
  --input_path "$sparse/0" \
  --output_path "$dense" \
  --output_type COLMAP

colmap patch_match_stereo \
  --workspace_path "$dense" \
  --workspace_format COLMAP \
  --PatchMatchStereo.geom_consistency true

colmap stereo_fusion \
  --workspace_path "$dense" \
  --workspace_format COLMAP \
  --input_type geometric \
  --output_path "$dense/fused.ply"

colmap poisson_mesher \
  --input_path "$dense/fused.ply" \
  --output_path "$dense/meshed-poisson.ply"
