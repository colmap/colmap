#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_alpha_numerator_denominator(
    float* SimpleRadialExtraCalib_p_kp1,
    unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
    float* SimpleRadialExtraCalib_r_k,
    unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
    float* SimpleRadialExtraCalib_w,
    unsigned int SimpleRadialExtraCalib_w_num_alloc,
    float* const SimpleRadialExtraCalib_total_ag,
    float* const SimpleRadialExtraCalib_total_ac,
    size_t problem_size);

}  // namespace caspar