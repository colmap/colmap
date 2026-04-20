#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_alpha_denumerator_or_beta_nummerator(
    float* Point_p_kp1,
    unsigned int Point_p_kp1_num_alloc,
    float* Point_w,
    unsigned int Point_w_num_alloc,
    float* const Point_out,
    size_t problem_size);

}  // namespace caspar