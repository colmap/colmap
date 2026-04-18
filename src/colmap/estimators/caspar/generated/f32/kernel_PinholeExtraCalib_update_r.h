#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_r(
    float* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    float* PinholeExtraCalib_w,
    unsigned int PinholeExtraCalib_w_num_alloc,
    const float* const negalpha,
    float* out_PinholeExtraCalib_r_kp1,
    unsigned int out_PinholeExtraCalib_r_kp1_num_alloc,
    float* const out_PinholeExtraCalib_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar