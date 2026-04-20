#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_update_Mp(
    double* SimpleRadialPrincipalPoint_r_k,
    unsigned int SimpleRadialPrincipalPoint_r_k_num_alloc,
    double* SimpleRadialPrincipalPoint_Mp,
    unsigned int SimpleRadialPrincipalPoint_Mp_num_alloc,
    const double* const beta,
    double* out_SimpleRadialPrincipalPoint_Mp_kp1,
    unsigned int out_SimpleRadialPrincipalPoint_Mp_kp1_num_alloc,
    double* out_SimpleRadialPrincipalPoint_w,
    unsigned int out_SimpleRadialPrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar