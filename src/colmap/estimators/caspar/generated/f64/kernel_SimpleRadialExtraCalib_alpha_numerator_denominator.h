#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_alpha_numerator_denominator(
    double* SimpleRadialExtraCalib_p_kp1,
    unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
    double* SimpleRadialExtraCalib_r_k,
    unsigned int SimpleRadialExtraCalib_r_k_num_alloc,
    double* SimpleRadialExtraCalib_w,
    unsigned int SimpleRadialExtraCalib_w_num_alloc,
    double* const SimpleRadialExtraCalib_total_ag,
    double* const SimpleRadialExtraCalib_total_ac,
    size_t problem_size);

}  // namespace caspar