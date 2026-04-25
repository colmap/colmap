#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_alpha_denumerator_or_beta_nummerator(
    double* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    double* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    double* const PinholePose_out,
    size_t problem_size);

}  // namespace caspar