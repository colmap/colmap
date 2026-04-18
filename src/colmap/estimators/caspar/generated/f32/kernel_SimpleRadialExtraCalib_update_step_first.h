#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_update_step_first(
    float* SimpleRadialExtraCalib_p_kp1,
    unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
    const float* const alpha,
    float* out_SimpleRadialExtraCalib_step_kp1,
    unsigned int out_SimpleRadialExtraCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar