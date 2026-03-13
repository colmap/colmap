#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_step(float* PinholeCalib_step_k,
                              unsigned int PinholeCalib_step_k_num_alloc,
                              float* PinholeCalib_p_kp1,
                              unsigned int PinholeCalib_p_kp1_num_alloc,
                              const float* const alpha,
                              float* out_PinholeCalib_step_kp1,
                              unsigned int out_PinholeCalib_step_kp1_num_alloc,
                              size_t problem_size);

}  // namespace caspar