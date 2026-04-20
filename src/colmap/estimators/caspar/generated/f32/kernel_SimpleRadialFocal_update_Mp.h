#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_update_Mp(
    float* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    float* SimpleRadialFocal_Mp,
    unsigned int SimpleRadialFocal_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocal_Mp_kp1,
    unsigned int out_SimpleRadialFocal_Mp_kp1_num_alloc,
    float* out_SimpleRadialFocal_w,
    unsigned int out_SimpleRadialFocal_w_num_alloc,
    size_t problem_size);

}  // namespace caspar