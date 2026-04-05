#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_update_step(
    float* SimpleRadialCalib_step_k,
    unsigned int SimpleRadialCalib_step_k_num_alloc,
    float* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    const float* const alpha,
    float* out_SimpleRadialCalib_step_kp1,
    unsigned int out_SimpleRadialCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar