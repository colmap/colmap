#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_update_Mp(
    float* SimpleRadialExtraCalib_r_k,
    unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
    float* SimpleRadialExtraCalib_Mp,
    unsigned int SimpleRadialExtraCalib_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialExtraCalib_Mp_kp1,
    unsigned int out_SimpleRadialExtraCalib_Mp_kp1_num_alloc,
    float* out_SimpleRadialExtraCalib_w,
    unsigned int out_SimpleRadialExtraCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar