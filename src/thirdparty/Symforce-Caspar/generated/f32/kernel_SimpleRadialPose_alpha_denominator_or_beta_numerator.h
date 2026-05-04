#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPoseAlphaDenominatorOrBetaNumerator(
    float* SimpleRadialPose_p_kp1,
    unsigned int SimpleRadialPose_p_kp1_num_alloc,
    float* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    float* const SimpleRadialPose_out,
    size_t problem_size);

}  // namespace caspar