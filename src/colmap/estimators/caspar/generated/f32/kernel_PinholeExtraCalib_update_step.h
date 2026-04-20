#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_step(
    float* PinholeExtraCalib_step_k,
    unsigned int PinholeExtraCalib_step_k_num_alloc,
    float* PinholeExtraCalib_p_kp1,
    unsigned int PinholeExtraCalib_p_kp1_num_alloc,
    const float* const alpha,
    float* out_PinholeExtraCalib_step_kp1,
    unsigned int out_PinholeExtraCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar