#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionPredDecreaseTimesTwo(
    float* SimpleRadialFocalAndDistortion_step,
    unsigned int SimpleRadialFocalAndDistortion_step_num_alloc,
    float* SimpleRadialFocalAndDistortion_precond_diag,
    unsigned int SimpleRadialFocalAndDistortion_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialFocalAndDistortion_njtr,
    unsigned int SimpleRadialFocalAndDistortion_njtr_num_alloc,
    float* const out_SimpleRadialFocalAndDistortion_pred_dec,
    size_t problem_size);

}  // namespace caspar