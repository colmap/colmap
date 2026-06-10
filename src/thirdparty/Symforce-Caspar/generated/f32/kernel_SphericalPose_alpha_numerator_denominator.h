#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseAlphaNumeratorDenominator(
    float* SphericalPose_p_kp1,
    unsigned int SphericalPose_p_kp1_num_alloc,
    float* SphericalPose_r_k,
    unsigned int SphericalPose_r_k_num_alloc,
    float* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    float* const SphericalPose_total_ag,
    float* const SphericalPose_total_ac,
    size_t problem_size);

}  // namespace caspar