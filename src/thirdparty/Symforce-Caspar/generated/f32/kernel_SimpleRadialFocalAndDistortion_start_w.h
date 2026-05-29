#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionStartW(
    float* SimpleRadialFocalAndDistortion_precond_diag,
    unsigned int SimpleRadialFocalAndDistortion_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialFocalAndDistortion_p,
    unsigned int SimpleRadialFocalAndDistortion_p_num_alloc,
    float* out_SimpleRadialFocalAndDistortion_w,
    unsigned int out_SimpleRadialFocalAndDistortion_w_num_alloc,
    size_t problem_size);

}  // namespace caspar