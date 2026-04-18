#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_start_w_contribute(
    float* SimpleRadialExtraCalib_precond_diag,
    unsigned int SimpleRadialExtraCalib_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialExtraCalib_p,
    unsigned int SimpleRadialExtraCalib_p_num_alloc,
    float* out_SimpleRadialExtraCalib_w,
    unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar