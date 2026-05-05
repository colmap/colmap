#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateStep(
    float* SimpleRadialFocalAndDistortion_step_k,
    unsigned int SimpleRadialFocalAndDistortion_step_k_num_alloc,
    float* SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    const float* const alpha,
    float* out_SimpleRadialFocalAndDistortion_step_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar