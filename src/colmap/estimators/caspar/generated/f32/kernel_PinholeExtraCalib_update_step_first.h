#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_step_first(
    float* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    const float* const alpha,
    float* out_PinholeExtraCalib_step_kp1,
    unsigned int out_PinholeExtraCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar