#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseAlphaDenominatorOrBetaNumerator(
    float* SphericalPose_p_kp1,
    unsigned int SphericalPose_p_kp1_num_alloc,
    float* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    float* const SphericalPose_out,
    size_t problem_size);

}  // namespace caspar