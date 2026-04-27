#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_alpha_numerator_denominator(
    double* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    double* PinholePose_r_k,
    unsigned int PinholePose_r_k_num_alloc,
    double* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    double* const PinholePose_total_ag,
    double* const PinholePose_total_ac,
    size_t problem_size);

}  // namespace caspar