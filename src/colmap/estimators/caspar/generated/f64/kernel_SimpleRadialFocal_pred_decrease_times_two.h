#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_pred_decrease_times_two(
    double* SimpleRadialFocal_step,
    unsigned int SimpleRadialFocal_step_num_alloc,
    double* SimpleRadialFocal_precond_diag,
    unsigned int SimpleRadialFocal_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialFocal_njtr,
    unsigned int SimpleRadialFocal_njtr_num_alloc,
    double* const out_SimpleRadialFocal_pred_dec,
    size_t problem_size);

}  // namespace caspar