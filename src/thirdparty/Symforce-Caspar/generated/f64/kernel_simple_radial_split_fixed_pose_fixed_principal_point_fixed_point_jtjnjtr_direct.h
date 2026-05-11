#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialSplitFixedPoseFixedPrincipalPointFixedPointJtjnjtrDirect(
    double* focal_and_distortion_njtr,
    unsigned int focal_and_distortion_njtr_num_alloc,
    SharedIndex* focal_and_distortion_njtr_indices,
    double* focal_and_distortion_jac,
    unsigned int focal_and_distortion_jac_num_alloc,
    double* const out_focal_and_distortion_njtr,
    unsigned int out_focal_and_distortion_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar