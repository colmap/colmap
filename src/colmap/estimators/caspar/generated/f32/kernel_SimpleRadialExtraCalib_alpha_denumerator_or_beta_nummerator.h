#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_alpha_denumerator_or_beta_nummerator(
    float* SimpleRadialExtraCalib_p_kp1,
    unsigned int SimpleRadialExtraCalib_p_kp1_num_alloc,
    float* SimpleRadialExtraCalib_w,
    unsigned int SimpleRadialExtraCalib_w_num_alloc,
    float* const SimpleRadialExtraCalib_out,
    size_t problem_size);

}  // namespace caspar