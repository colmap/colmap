#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_r(float* PinholeCalib_r_k,
                           unsigned int PinholeCalib_r_k_num_alloc,
                           float* PinholeCalib_w,
                           unsigned int PinholeCalib_w_num_alloc,
                           const float* const negalpha,
                           float* out_PinholeCalib_r_kp1,
                           unsigned int out_PinholeCalib_r_kp1_num_alloc,
                           float* const out_PinholeCalib_r_kp1_norm2_tot,
                           size_t problem_size);

}  // namespace caspar