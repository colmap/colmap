#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
    float *pose_njtr, unsigned int pose_njtr_num_alloc,
    SharedIndex *pose_njtr_indices, float *pose_jac,
    unsigned int pose_jac_num_alloc, float *focal_and_distortion_njtr,
    unsigned int focal_and_distortion_njtr_num_alloc,
    SharedIndex *focal_and_distortion_njtr_indices,
    float *focal_and_distortion_jac,
    unsigned int focal_and_distortion_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float *const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc, size_t problem_size);

} // namespace caspar