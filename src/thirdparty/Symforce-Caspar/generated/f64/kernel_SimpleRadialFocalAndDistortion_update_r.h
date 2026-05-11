#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateR(
    double *SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    double *SimpleRadialFocalAndDistortion_w,
    unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
    const double *const negalpha,
    double *out_SimpleRadialFocalAndDistortion_r_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_r_kp1_num_alloc,
    double *const out_SimpleRadialFocalAndDistortion_r_kp1_norm2_tot,
    size_t problem_size);

} // namespace caspar