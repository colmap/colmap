#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_update_Mp(
    float* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    float* SimpleRadialPose_Mp,
    unsigned int SimpleRadialPose_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialPose_Mp_kp1,
    unsigned int out_SimpleRadialPose_Mp_kp1_num_alloc,
    float* out_SimpleRadialPose_w,
    unsigned int out_SimpleRadialPose_w_num_alloc,
    size_t problem_size);

}  // namespace caspar