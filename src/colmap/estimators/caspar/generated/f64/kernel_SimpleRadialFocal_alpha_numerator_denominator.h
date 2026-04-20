#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_alpha_numerator_denominator(
    double* SimpleRadialFocal_p_kp1,
    unsigned int SimpleRadialFocal_p_kp1_num_alloc,
    double* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    double* SimpleRadialFocal_w,
    unsigned int SimpleRadialFocal_w_num_alloc,
    double* const SimpleRadialFocal_total_ag,
    double* const SimpleRadialFocal_total_ac,
    size_t problem_size);

}  // namespace caspar