#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointScore(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *pixel, unsigned int pixel_num_alloc, float *focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *point,
    unsigned int point_num_alloc, float *const out_rTr, size_t problem_size);

} // namespace caspar