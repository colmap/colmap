#!/bin/bash

frame=038512
DATASET_PATH_IMG='/home/gaini/capstone/dataset' # path to the dataset with images
DATASET_PATH='/home/gaini/capstone/known_intrinsics' # path to the folder with the database created by colmap with extracted features and etc

echo $frame
while read p; do
  IFS=: read -r id params <<< "$p"

  colmap feature_extractor --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH_IMG/frames/$id \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model=PINHOLE \
  --ImageReader.camera_params $params

  colmap exhaustive_matcher --database_path $DATASET_PATH/database.db

  mkdir -p $DATASET_PATH/sparse

  colmap mapper \
  --database_path $DATASET_PATH/database.db \
  --image_path $DATASET_PATH_IMG/frames \
  --output_path $DATASET_PATH/sparse \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.ba_refine_focal_length false 0 \
  --Mapper.ba_refine_extra_params 0

  done <intrinsics.txt