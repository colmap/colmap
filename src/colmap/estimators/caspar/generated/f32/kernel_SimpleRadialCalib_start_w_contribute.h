#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_start_w_contribute(
    float* SimpleRadialCalib_precond_diag,
    unsigned int SimpleRadialCalib_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialCalib_p,
    unsigned int SimpleRadialCalib_p_num_alloc,
    float* out_SimpleRadialCalib_w,
    unsigned int out_SimpleRadialCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar