#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_Mp(float* PinholePose_r_k,
                           unsigned int PinholePose_r_k_num_alloc,
                           float* PinholePose_Mp,
                           unsigned int PinholePose_Mp_num_alloc,
                           const float* const beta,
                           float* out_PinholePose_Mp_kp1,
                           unsigned int out_PinholePose_Mp_kp1_num_alloc,
                           float* out_PinholePose_w,
                           unsigned int out_PinholePose_w_num_alloc,
                           size_t problem_size);

}  // namespace caspar