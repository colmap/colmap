#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialFocal_p_kp1,
    unsigned int SimpleRadialFocal_p_kp1_num_alloc,
    double* SimpleRadialFocal_w,
    unsigned int SimpleRadialFocal_w_num_alloc,
    double* const SimpleRadialFocal_out,
    size_t problem_size);

}  // namespace caspar