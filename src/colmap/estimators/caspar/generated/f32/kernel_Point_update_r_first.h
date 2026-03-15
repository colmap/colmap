#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_update_r_first(float* Point_r_k,
                          unsigned int Point_r_k_num_alloc,
                          float* Point_w,
                          unsigned int Point_w_num_alloc,
                          const float* const negalpha,
                          float* out_Point_r_kp1,
                          unsigned int out_Point_r_kp1_num_alloc,
                          float* const out_Point_r_0_norm2_tot,
                          float* const out_Point_r_kp1_norm2_tot,
                          size_t problem_size);

}  // namespace caspar