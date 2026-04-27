#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_alpha_numerator_denominator(
    float* SimpleRadialPose_p_kp1,
    unsigned int SimpleRadialPose_p_kp1_num_alloc,
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    float* const SimpleRadialPose_total_ag,
    float* const SimpleRadialPose_total_ac,
    size_t problem_size);

}  // namespace caspar