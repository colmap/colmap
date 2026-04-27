#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_pred_decrease_times_two(
    double* PinholePose_step,
    unsigned int PinholePose_step_num_alloc,
    double* PinholePose_precond_diag,
    unsigned int PinholePose_precond_diag_num_alloc,
    const double* const diag,
    double* PinholePose_njtr,
    unsigned int PinholePose_njtr_num_alloc,
    double* const out_PinholePose_pred_dec,
    size_t problem_size);

}  // namespace caspar