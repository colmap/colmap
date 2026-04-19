#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_update_r_first(
    double* SimpleRadialPrincipalPoint_r_k,
    unsigned int SimpleRadialPrincipalPoint_r_k_num_alloc,
    double* SimpleRadialPrincipalPoint_w,
    unsigned int SimpleRadialPrincipalPoint_w_num_alloc,
    const double* const negalpha,
    double* out_SimpleRadialPrincipalPoint_r_kp1,
    unsigned int out_SimpleRadialPrincipalPoint_r_kp1_num_alloc,
    double* const out_SimpleRadialPrincipalPoint_r_0_norm2_tot,
    double* const out_SimpleRadialPrincipalPoint_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar