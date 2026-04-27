#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_r_first(double* PinholePose_r_k,
                                unsigned int PinholePose_r_k_num_alloc,
                                double* PinholePose_w,
                                unsigned int PinholePose_w_num_alloc,
                                const double* const negalpha,
                                double* out_PinholePose_r_kp1,
                                unsigned int out_PinholePose_r_kp1_num_alloc,
                                double* const out_PinholePose_r_0_norm2_tot,
                                double* const out_PinholePose_r_kp1_norm2_tot,
                                size_t problem_size);

}  // namespace caspar