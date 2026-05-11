#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointJtjnjtrDirect(
    float *focal_and_distortion_njtr,
    unsigned int focal_and_distortion_njtr_num_alloc,
    SharedIndex *focal_and_distortion_njtr_indices,
    float *focal_and_distortion_jac,
    unsigned int focal_and_distortion_jac_num_alloc,
    float *const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc, size_t problem_size);

} // namespace caspar