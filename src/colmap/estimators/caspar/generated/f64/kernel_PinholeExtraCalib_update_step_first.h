#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_step_first(
    double* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    const double* const alpha,
    double* out_PinholeExtraCalib_step_kp1,
    unsigned int out_PinholeExtraCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar