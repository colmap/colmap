#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_update_p(
    float* SimpleRadialPrincipalPoint_z,
    unsigned int SimpleRadialPrincipalPoint_z_num_alloc,
    float* SimpleRadialPrincipalPoint_p_k,
    unsigned int SimpleRadialPrincipalPoint_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialPrincipalPoint_p_kp1,
    unsigned int out_SimpleRadialPrincipalPoint_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar