#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_alpha_numerator_denominator(
    double* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    double* PinholeExtraCalib_r_k,
    unsigned int PinholeExtraCalib_r_k_num_alloc,
    double* PinholeExtraCalib_w,
    unsigned int PinholeExtraCalib_w_num_alloc,
    double* const PinholeExtraCalib_total_ag,
    double* const PinholeExtraCalib_total_ac,
    size_t problem_size);

}  // namespace caspar