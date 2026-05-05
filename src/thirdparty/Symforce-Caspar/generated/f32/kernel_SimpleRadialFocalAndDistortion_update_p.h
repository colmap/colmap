#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateP(
    float* SimpleRadialFocalAndDistortion_z,
    unsigned int SimpleRadialFocalAndDistortion_z_num_alloc,
    float* SimpleRadialFocalAndDistortion_p_k,
    unsigned int SimpleRadialFocalAndDistortion_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar