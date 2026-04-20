#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_step_first(
    double* PinholeCalib_p_kp1,
    unsigned int PinholeCalib_p_kp1_num_alloc,
    const double* const alpha,
    double* out_PinholeCalib_step_kp1,
    unsigned int out_PinholeCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar