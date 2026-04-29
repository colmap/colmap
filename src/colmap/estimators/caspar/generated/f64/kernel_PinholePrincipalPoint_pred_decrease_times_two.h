#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPointPredDecreaseTimesTwo(
    double* PinholePrincipalPoint_step,
    unsigned int PinholePrincipalPoint_step_num_alloc,
    double* PinholePrincipalPoint_precond_diag,
    unsigned int PinholePrincipalPoint_precond_diag_num_alloc,
    const double* const diag,
    double* PinholePrincipalPoint_njtr,
    unsigned int PinholePrincipalPoint_njtr_num_alloc,
    double* const out_PinholePrincipalPoint_pred_dec,
    size_t problem_size);

}  // namespace caspar