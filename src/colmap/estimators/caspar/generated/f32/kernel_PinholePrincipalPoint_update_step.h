#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_update_step(
    float* PinholePrincipalPoint_step_k,
    unsigned int PinholePrincipalPoint_step_k_num_alloc,
    float* PinholePrincipalPoint_p_kp1,
    unsigned int PinholePrincipalPoint_p_kp1_num_alloc,
    const float* const alpha,
    float* out_PinholePrincipalPoint_step_kp1,
    unsigned int out_PinholePrincipalPoint_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar