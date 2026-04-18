#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_pred_decrease_times_two(
    float* PinholeExtraCalib_step,
    unsigned int PinholeExtraCalib_step_num_alloc,
    float* PinholeExtraCalib_precond_diag,
    unsigned int PinholeExtraCalib_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeExtraCalib_njtr,
    unsigned int PinholeExtraCalib_njtr_num_alloc,
    float* const out_PinholeExtraCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar