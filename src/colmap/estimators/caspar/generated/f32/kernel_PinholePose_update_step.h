#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePose_update_step(float* PinholePose_step_k,
                             unsigned int PinholePose_step_k_num_alloc,
                             float* PinholePose_p_kp1,
                             unsigned int PinholePose_p_kp1_num_alloc,
                             const float* const alpha,
                             float* out_PinholePose_step_kp1,
                             unsigned int out_PinholePose_step_kp1_num_alloc,
                             size_t problem_size);

}  // namespace caspar