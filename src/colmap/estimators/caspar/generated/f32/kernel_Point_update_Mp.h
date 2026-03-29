#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_update_Mp(float* Point_r_k,
                     unsigned int Point_r_k_num_alloc,
                     float* Point_Mp,
                     unsigned int Point_Mp_num_alloc,
                     const float* const beta,
                     float* out_Point_Mp_kp1,
                     unsigned int out_Point_Mp_kp1_num_alloc,
                     float* out_Point_w,
                     unsigned int out_Point_w_num_alloc,
                     size_t problem_size);

}  // namespace caspar