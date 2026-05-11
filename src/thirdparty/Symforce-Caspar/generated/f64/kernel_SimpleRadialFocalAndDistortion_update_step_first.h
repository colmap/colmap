#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateStepFirst(
    double* SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialFocalAndDistortion_step_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar