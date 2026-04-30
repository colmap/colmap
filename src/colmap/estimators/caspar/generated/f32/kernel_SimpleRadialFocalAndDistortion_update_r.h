#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateR(
    float* SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    float* SimpleRadialFocalAndDistortion_w,
    unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialFocalAndDistortion_r_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_r_kp1_num_alloc,
    float* const out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar