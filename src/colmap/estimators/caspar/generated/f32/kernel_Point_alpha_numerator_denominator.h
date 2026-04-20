#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Point_alpha_numerator_denominator(float* Point_p_kp1,
                                       unsigned int Point_p_kp1_num_alloc,
                                       float* Point_r_k,
                                       unsigned int Point_r_k_num_alloc,
                                       float* Point_w,
                                       unsigned int Point_w_num_alloc,
                                       float* const Point_total_ag,
                                       float* const Point_total_ac,
                                       size_t problem_size);

}  // namespace caspar