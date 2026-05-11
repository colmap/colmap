#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialSplitFixedPrincipalPointScore(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *focal_and_distortion, unsigned int focal_and_distortion_num_alloc,
    SharedIndex *focal_and_distortion_indices, double *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, double *pixel,
    unsigned int pixel_num_alloc, double *principal_point,
    unsigned int principal_point_num_alloc, double *const out_rTr,
    size_t problem_size);

} // namespace caspar