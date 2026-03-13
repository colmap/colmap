#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_update_r(double* Point_r_k,
                    unsigned int Point_r_k_num_alloc,
                    double* Point_w,
                    unsigned int Point_w_num_alloc,
                    const double* const negalpha,
                    double* out_Point_r_kp1,
                    unsigned int out_Point_r_kp1_num_alloc,
                    double* const out_Point_r_kp1_norm2_tot,
                    size_t problem_size);

}  // namespace caspar