#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialPrincipalPoint_p_kp1,
    unsigned int SimpleRadialPrincipalPoint_p_kp1_num_alloc,
    double* SimpleRadialPrincipalPoint_w,
    unsigned int SimpleRadialPrincipalPoint_w_num_alloc,
    double* const SimpleRadialPrincipalPoint_out,
    size_t problem_size);

}  // namespace caspar