#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_update_r(
    float* SimpleRadialPrincipalPoint_r_k,
    unsigned int SimpleRadialPrincipalPoint_r_k_num_alloc,
    float* SimpleRadialPrincipalPoint_w,
    unsigned int SimpleRadialPrincipalPoint_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialPrincipalPoint_r_kp1,
    unsigned int out_SimpleRadialPrincipalPoint_r_kp1_num_alloc,
    float* const out_SimpleRadialPrincipalPoint_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar