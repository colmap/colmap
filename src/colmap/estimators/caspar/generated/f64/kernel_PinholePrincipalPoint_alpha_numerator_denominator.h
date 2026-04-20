#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_alpha_numerator_denominator(
    double* PinholePrincipalPoint_p_kp1,
    unsigned int PinholePrincipalPoint_p_kp1_num_alloc,
    double* PinholePrincipalPoint_r_k,
    unsigned int PinholePrincipalPoint_r_k_num_alloc,
    double* PinholePrincipalPoint_w,
    unsigned int PinholePrincipalPoint_w_num_alloc,
    double* const PinholePrincipalPoint_total_ag,
    double* const PinholePrincipalPoint_total_ac,
    size_t problem_size);

}  // namespace caspar