#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_start_w(
    float* SimpleRadialFocal_precond_diag,
    unsigned int SimpleRadialFocal_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialFocal_p,
    unsigned int SimpleRadialFocal_p_num_alloc,
    float* out_SimpleRadialFocal_w,
    unsigned int out_SimpleRadialFocal_w_num_alloc,
    size_t problem_size);

}  // namespace caspar