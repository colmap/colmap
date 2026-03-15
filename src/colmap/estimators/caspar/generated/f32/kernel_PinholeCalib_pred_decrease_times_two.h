#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_pred_decrease_times_two(
    float* PinholeCalib_step,
    unsigned int PinholeCalib_step_num_alloc,
    float* PinholeCalib_precond_diag,
    unsigned int PinholeCalib_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeCalib_njtr,
    unsigned int PinholeCalib_njtr_num_alloc,
    float* const out_PinholeCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar