#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_update_step(
    double* SimpleRadialFocal_step_k,
    unsigned int SimpleRadialFocal_step_k_num_alloc,
    double* SimpleRadialFocal_p_kp1,
    unsigned int SimpleRadialFocal_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialFocal_step_kp1,
    unsigned int out_SimpleRadialFocal_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar