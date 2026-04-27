#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_update_step(
    float* SimpleRadialPrincipalPoint_step_k,
    unsigned int SimpleRadialPrincipalPoint_step_k_num_alloc,
    float* SimpleRadialPrincipalPoint_p_kp1,
    unsigned int SimpleRadialPrincipalPoint_p_kp1_num_alloc,
    const float* const alpha,
    float* out_SimpleRadialPrincipalPoint_step_kp1,
    unsigned int out_SimpleRadialPrincipalPoint_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar