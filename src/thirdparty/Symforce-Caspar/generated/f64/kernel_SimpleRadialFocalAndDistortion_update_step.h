#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateStep(
    double *SimpleRadialFocalAndDistortion_step_k,
    unsigned int SimpleRadialFocalAndDistortion_step_k_num_alloc,
    double *SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    const double *const alpha,
    double *out_SimpleRadialFocalAndDistortion_step_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_step_kp1_num_alloc,
    size_t problem_size);

} // namespace caspar