#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_update_step_first(
    double* SimpleRadialPrincipalPoint_p_kp1,
    unsigned int SimpleRadialPrincipalPoint_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialPrincipalPoint_step_kp1,
    unsigned int out_SimpleRadialPrincipalPoint_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar