#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitFixedPoseFixedPrincipalPointScore(
    double* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* principal_point,
    unsigned int principal_point_num_alloc,
    double* const out_rTr,
    size_t problem_size);

}  // namespace caspar