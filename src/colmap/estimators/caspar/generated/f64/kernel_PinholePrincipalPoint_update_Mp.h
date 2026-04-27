#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_update_Mp(
    double* PinholePrincipalPoint_r_k,
    unsigned int PinholePrincipalPoint_r_k_num_alloc,
    double* PinholePrincipalPoint_Mp,
    unsigned int PinholePrincipalPoint_Mp_num_alloc,
    const double* const beta,
    double* out_PinholePrincipalPoint_Mp_kp1,
    unsigned int out_PinholePrincipalPoint_Mp_kp1_num_alloc,
    double* out_PinholePrincipalPoint_w,
    unsigned int out_PinholePrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar