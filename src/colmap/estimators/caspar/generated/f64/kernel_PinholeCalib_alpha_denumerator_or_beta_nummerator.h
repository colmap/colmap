#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_alpha_denumerator_or_beta_nummerator(
    double* PinholeCalib_p_kp1,
    unsigned int PinholeCalib_p_kp1_num_alloc,
    double* PinholeCalib_w,
    unsigned int PinholeCalib_w_num_alloc,
    double* const PinholeCalib_out,
    size_t problem_size);

}  // namespace caspar