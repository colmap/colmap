#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_update_p(
    float* SimpleRadialPose_z,
    unsigned int SimpleRadialPose_z_num_alloc,
    float* SimpleRadialPose_p_k,
    unsigned int SimpleRadialPose_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialPose_p_kp1,
    unsigned int out_SimpleRadialPose_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar