#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_alpha_denumerator_or_beta_nummerator(
    double* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    double* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    double* const SimpleRadialCalib_out,
    size_t problem_size);

}  // namespace caspar