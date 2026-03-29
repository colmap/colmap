#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_alpha_numerator_denominator(
    float* PinholeCalib_p_kp1,
    unsigned int PinholeCalib_p_kp1_num_alloc,
    float* PinholeCalib_r_k,
    unsigned int PinholeCalib_r_k_num_alloc,
    float* PinholeCalib_w,
    unsigned int PinholeCalib_w_num_alloc,
    float* const PinholeCalib_total_ag,
    float* const PinholeCalib_total_ac,
    size_t problem_size);

}  // namespace caspar