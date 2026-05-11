#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionPredDecreaseTimesTwo(
    double* SimpleRadialFocalAndDistortion_step,
    unsigned int SimpleRadialFocalAndDistortion_step_num_alloc,
    double* SimpleRadialFocalAndDistortion_precond_diag,
    unsigned int SimpleRadialFocalAndDistortion_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialFocalAndDistortion_njtr,
    unsigned int SimpleRadialFocalAndDistortion_njtr_num_alloc,
    double* const out_SimpleRadialFocalAndDistortion_pred_dec,
    size_t problem_size);

}  // namespace caspar