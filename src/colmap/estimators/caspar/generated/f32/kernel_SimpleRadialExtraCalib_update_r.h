#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_update_r(
    float* SimpleRadialExtraCalib_r_k,
    unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
    float* SimpleRadialExtraCalib_w,
    unsigned int SimpleRadialExtraCalib_w_num_alloc,
    const float* const negalpha,
    float* out_SimpleRadialExtraCalib_r_kp1,
    unsigned int out_SimpleRadialExtraCalib_r_kp1_num_alloc,
    float* const out_SimpleRadialExtraCalib_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar