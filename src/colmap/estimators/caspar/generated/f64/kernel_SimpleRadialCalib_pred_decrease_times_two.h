#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_pred_decrease_times_two(
    double* SimpleRadialCalib_step,
    unsigned int SimpleRadialCalib_step_num_alloc,
    double* SimpleRadialCalib_precond_diag,
    unsigned int SimpleRadialCalib_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialCalib_njtr,
    unsigned int SimpleRadialCalib_njtr_num_alloc,
    double* const out_SimpleRadialCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar