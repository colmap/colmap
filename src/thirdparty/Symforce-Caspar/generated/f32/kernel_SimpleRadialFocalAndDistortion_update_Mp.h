#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateMp(
    float* SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    float* SimpleRadialFocalAndDistortion_Mp,
    unsigned int SimpleRadialFocalAndDistortion_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocalAndDistortion_Mp_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_Mp_kp1_num_alloc,
    float* out_SimpleRadialFocalAndDistortion_w,
    unsigned int out_SimpleRadialFocalAndDistortion_w_num_alloc,
    size_t problem_size);

}  // namespace caspar