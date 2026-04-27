#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_update_p(
    float* PinholePrincipalPoint_z,
    unsigned int PinholePrincipalPoint_z_num_alloc,
    float* PinholePrincipalPoint_p_k,
    unsigned int PinholePrincipalPoint_p_k_num_alloc,
    const float* const beta,
    float* out_PinholePrincipalPoint_p_kp1,
    unsigned int out_PinholePrincipalPoint_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar