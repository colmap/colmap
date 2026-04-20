#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_update_r(
    double* PinholePrincipalPoint_r_k,
    unsigned int PinholePrincipalPoint_r_k_num_alloc,
    double* PinholePrincipalPoint_w,
    unsigned int PinholePrincipalPoint_w_num_alloc,
    const double* const negalpha,
    double* out_PinholePrincipalPoint_r_kp1,
    unsigned int out_PinholePrincipalPoint_r_kp1_num_alloc,
    double* const out_PinholePrincipalPoint_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar