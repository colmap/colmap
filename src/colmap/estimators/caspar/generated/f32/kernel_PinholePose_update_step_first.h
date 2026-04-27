#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_step_first(
    float* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    const float* const alpha,
    float* out_PinholePose_step_kp1,
    unsigned int out_PinholePose_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar