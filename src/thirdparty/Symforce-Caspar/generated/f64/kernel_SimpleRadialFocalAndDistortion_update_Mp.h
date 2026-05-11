#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialFocalAndDistortionUpdateMp(
    double *SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    double *SimpleRadialFocalAndDistortion_Mp,
    unsigned int SimpleRadialFocalAndDistortion_Mp_num_alloc,
    const double *const beta, double *out_SimpleRadialFocalAndDistortion_Mp_kp1,
    unsigned int out_SimpleRadialFocalAndDistortion_Mp_kp1_num_alloc,
    double *out_SimpleRadialFocalAndDistortion_w,
    unsigned int out_SimpleRadialFocalAndDistortion_w_num_alloc,
    size_t problem_size);

} // namespace caspar