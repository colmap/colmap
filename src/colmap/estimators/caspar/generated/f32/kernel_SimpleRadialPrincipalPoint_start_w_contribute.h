#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_start_w_contribute(
    float* SimpleRadialPrincipalPoint_precond_diag,
    unsigned int SimpleRadialPrincipalPoint_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialPrincipalPoint_p,
    unsigned int SimpleRadialPrincipalPoint_p_num_alloc,
    float* out_SimpleRadialPrincipalPoint_w,
    unsigned int out_SimpleRadialPrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar