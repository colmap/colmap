#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_pred_decrease_times_two(
    double* PinholeCalib_step,
    unsigned int PinholeCalib_step_num_alloc,
    double* PinholeCalib_precond_diag,
    unsigned int PinholeCalib_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeCalib_njtr,
    unsigned int PinholeCalib_njtr_num_alloc,
    double* const out_PinholeCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar