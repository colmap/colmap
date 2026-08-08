#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalFixedPointScore(double* pose,
                              unsigned int pose_num_alloc,
                              SharedIndex* pose_indices,
                              double* sensor_from_rig,
                              unsigned int sensor_from_rig_num_alloc,
                              double* wh,
                              unsigned int wh_num_alloc,
                              double* pixel,
                              unsigned int pixel_num_alloc,
                              double* point,
                              unsigned int point_num_alloc,
                              double* const out_rTr,
                              size_t problem_size);

}  // namespace caspar