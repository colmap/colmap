#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_Mp(double* PinholePose_r_k,
                           unsigned int PinholePose_r_k_num_alloc,
                           double* PinholePose_Mp,
                           unsigned int PinholePose_Mp_num_alloc,
                           const double* const beta,
                           double* out_PinholePose_Mp_kp1,
                           unsigned int out_PinholePose_Mp_kp1_num_alloc,
                           double* out_PinholePose_w,
                           unsigned int out_PinholePose_w_num_alloc,
                           size_t problem_size);

}  // namespace caspar