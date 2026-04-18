#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_alpha_numerator_denominator(
    float* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    float* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    float* PinholeExtraCalib_w,
    unsigned int PinholeExtraCalib_w_num_alloc,
    float* const PinholeExtraCalib_total_ag,
    float* const PinholeExtraCalib_total_ac,
    size_t problem_size);

}  // namespace caspar