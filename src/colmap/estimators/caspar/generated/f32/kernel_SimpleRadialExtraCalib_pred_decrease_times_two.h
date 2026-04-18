#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_pred_decrease_times_two(
    float* SimpleRadialExtraCalib_step,
    unsigned int SimpleRadialExtraCalib_step_num_alloc,
    float* SimpleRadialExtraCalib_precond_diag,
    unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialExtraCalib_njtr,
    unsigned int SimpleRadialExtraCalib_njtr_num_alloc,
    float* const out_SimpleRadialExtraCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar