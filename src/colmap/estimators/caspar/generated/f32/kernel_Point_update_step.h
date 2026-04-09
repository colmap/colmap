#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_update_step(float* Point_step_k,
                       unsigned int Point_step_k_num_alloc,
                       float* Point_p_kp1,
                       unsigned int Point_p_kp1_num_alloc,
                       const float* const alpha,
                       float* out_Point_step_kp1,
                       unsigned int out_Point_step_kp1_num_alloc,
                       size_t problem_size);

}  // namespace caspar