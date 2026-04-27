#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_r(float* PinholePose_r_k,
                          unsigned int PinholePose_r_k_num_alloc,
                          float* PinholePose_w,
                          unsigned int PinholePose_w_num_alloc,
                          const float* const negalpha,
                          float* out_PinholePose_r_kp1,
                          unsigned int out_PinholePose_r_kp1_num_alloc,
                          float* const out_PinholePose_r_kp1_norm2_tot,
                          size_t problem_size);

}  // namespace caspar