#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_alpha_numerator_denominator(
    float* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    float* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    float* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    float* const SimpleRadialCalib_total_ag,
    float* const SimpleRadialCalib_total_ac,
    size_t problem_size);

}  // namespace caspar