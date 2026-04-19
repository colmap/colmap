#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_alpha_denumerator_or_beta_nummerator(
    double* PinholePrincipalPoint_p_kp1,
    unsigned int PinholePrincipalPoint_p_kp1_num_alloc,
    double* PinholePrincipalPoint_w,
    unsigned int PinholePrincipalPoint_w_num_alloc,
    double* const PinholePrincipalPoint_out,
    size_t problem_size);

}  // namespace caspar