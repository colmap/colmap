#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_pred_decrease_times_two(
    float* SimpleRadialPose_step,
    unsigned int SimpleRadialPose_step_num_alloc,
    float* SimpleRadialPose_precond_diag,
    unsigned int SimpleRadialPose_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialPose_njtr,
    unsigned int SimpleRadialPose_njtr_num_alloc,
    float* const out_SimpleRadialPose_pred_dec,
    size_t problem_size);

}  // namespace caspar