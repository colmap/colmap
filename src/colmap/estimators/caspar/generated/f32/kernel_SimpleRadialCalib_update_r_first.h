#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_update_r_first(
    float* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    float* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialCalib_r_kp1,
    unsigned int out_SimpleRadialCalib_r_kp1_num_alloc,
    float* const out_SimpleRadialCalib_r_0_norm2_tot,
    float* const out_SimpleRadialCalib_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar