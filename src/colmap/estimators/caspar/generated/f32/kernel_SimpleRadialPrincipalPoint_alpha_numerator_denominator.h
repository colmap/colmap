#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_alpha_numerator_denominator(
    float* SimpleRadialPrincipalPoint_p_kp1,
    unsigned int SimpleRadialPrincipalPoint_p_kp1_num_alloc,
    float* SimpleRadialPrincipalPoint_r_k,
    unsigned int SimpleRadialPrincipalPoint_r_k_num_alloc,
    float* SimpleRadialPrincipalPoint_w,
    unsigned int SimpleRadialPrincipalPoint_w_num_alloc,
    float* const SimpleRadialPrincipalPoint_total_ag,
    float* const SimpleRadialPrincipalPoint_total_ac,
    size_t problem_size);

}  // namespace caspar