#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_pred_decrease_times_two(
    float* SimpleRadialPrincipalPoint_step,
    unsigned int SimpleRadialPrincipalPoint_step_num_alloc,
    float* SimpleRadialPrincipalPoint_precond_diag,
    unsigned int SimpleRadialPrincipalPoint_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialPrincipalPoint_njtr,
    unsigned int SimpleRadialPrincipalPoint_njtr_num_alloc,
    float* const out_SimpleRadialPrincipalPoint_pred_dec,
    size_t problem_size);

}  // namespace caspar