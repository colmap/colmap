#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalFixedPointScore(float* pose,
                              unsigned int pose_num_alloc,
                              SharedIndex* pose_indices,
                              float* sensor_from_rig,
                              unsigned int sensor_from_rig_num_alloc,
                              float* wh,
                              unsigned int wh_num_alloc,
                              float* pixel,
                              unsigned int pixel_num_alloc,
                              float* point,
                              unsigned int point_num_alloc,
                              float* const out_rTr,
                              size_t problem_size);

}  // namespace caspar