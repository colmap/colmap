#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_pred_decrease_times_two(
    double* SimpleRadialPrincipalPoint_step,
    unsigned int SimpleRadialPrincipalPoint_step_num_alloc,
    double* SimpleRadialPrincipalPoint_precond_diag,
    unsigned int SimpleRadialPrincipalPoint_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialPrincipalPoint_njtr,
    unsigned int SimpleRadialPrincipalPoint_njtr_num_alloc,
    double* const out_SimpleRadialPrincipalPoint_pred_dec,
    size_t problem_size);

}  // namespace caspar