#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalFixedPoseScore(double* sensor_from_rig,
                             unsigned int sensor_from_rig_num_alloc,
                             double* wh,
                             unsigned int wh_num_alloc,
                             double* point,
                             unsigned int point_num_alloc,
                             SharedIndex* point_indices,
                             double* pixel,
                             unsigned int pixel_num_alloc,
                             double* pose,
                             unsigned int pose_num_alloc,
                             double* const out_rTr,
                             size_t problem_size);

}  // namespace caspar