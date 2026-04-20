#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_update_Mp(
    float* PinholePrincipalPoint_r_k,
    unsigned int PinholePrincipalPoint_r_k_num_alloc,
    float* PinholePrincipalPoint_Mp,
    unsigned int PinholePrincipalPoint_Mp_num_alloc,
    const float* const beta,
    float* out_PinholePrincipalPoint_Mp_kp1,
    unsigned int out_PinholePrincipalPoint_Mp_kp1_num_alloc,
    float* out_PinholePrincipalPoint_w,
    unsigned int out_PinholePrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar