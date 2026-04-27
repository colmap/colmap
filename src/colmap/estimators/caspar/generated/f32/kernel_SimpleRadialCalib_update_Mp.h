#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_update_Mp(
    float* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    float* SimpleRadialCalib_Mp,
    unsigned int SimpleRadialCalib_Mp_num_alloc,
    const float* const beta,
    float* out_SimpleRadialCalib_Mp_kp1,
    unsigned int out_SimpleRadialCalib_Mp_kp1_num_alloc,
    float* out_SimpleRadialCalib_w,
    unsigned int out_SimpleRadialCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar