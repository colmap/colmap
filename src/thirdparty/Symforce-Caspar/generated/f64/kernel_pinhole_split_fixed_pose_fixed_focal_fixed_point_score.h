#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void PinholeSplitFixedPoseFixedFocalFixedPointScore(
    double *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *pose, unsigned int pose_num_alloc,
    double *focal, unsigned int focal_num_alloc, double *point,
    unsigned int point_num_alloc, double *const out_rTr, size_t problem_size);

} // namespace caspar