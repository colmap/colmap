#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_update_Mp(double* Point_r_k,
                     unsigned int Point_r_k_num_alloc,
                     double* Point_Mp,
                     unsigned int Point_Mp_num_alloc,
                     const double* const beta,
                     double* out_Point_Mp_kp1,
                     unsigned int out_Point_Mp_kp1_num_alloc,
                     double* out_Point_w,
                     unsigned int out_Point_w_num_alloc,
                     size_t problem_size);

}  // namespace caspar