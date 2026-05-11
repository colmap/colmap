#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateP(
    double *SimpleRadialFocalAndDistortion_z,
    unsigned int SimpleRadialFocalAndDistortion_z_num_alloc,
    double *SimpleRadialFocalAndDistortion_p_k,
    unsigned int SimpleRadialFocalAndDistortion_p_k_num_alloc,
    const double *const beta, double *out_SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    size_t problem_size);

} // namespace caspar