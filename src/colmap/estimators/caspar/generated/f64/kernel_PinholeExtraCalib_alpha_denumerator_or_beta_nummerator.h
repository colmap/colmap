#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_alpha_denumerator_or_beta_nummerator(
    double* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    double* PinholeExtraCalib_w,
    unsigned int PinholeExtraCalib_w_num_alloc,
    double* const PinholeExtraCalib_out,
    size_t problem_size);

}  // namespace caspar