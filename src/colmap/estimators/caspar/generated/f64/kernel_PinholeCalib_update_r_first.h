#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_r_first(double* PinholeCalib_r_k,
                                 unsigned int PinholeCalib_r_k_num_alloc,
                                 double* PinholeCalib_w,
                                 unsigned int PinholeCalib_w_num_alloc,
                                 const double* const negalpha,
                                 double* out_PinholeCalib_r_kp1,
                                 unsigned int out_PinholeCalib_r_kp1_num_alloc,
                                 double* const out_PinholeCalib_r_0_norm2_tot,
                                 double* const out_PinholeCalib_r_kp1_norm2_tot,
                                 size_t problem_size);

}  // namespace caspar