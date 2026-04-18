#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_update_r(
    float* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    float* SimpleRadialFocal_w,
    unsigned int SimpleRadialFocal_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialFocal_r_kp1,
    unsigned int out_SimpleRadialFocal_r_kp1_num_alloc,
    float* const out_SimpleRadialFocal_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar