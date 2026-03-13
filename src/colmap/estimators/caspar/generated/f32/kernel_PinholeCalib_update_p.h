#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalib_update_p(float* PinholeCalib_z,
                           unsigned int PinholeCalib_z_num_alloc,
                           float* PinholeCalib_p_k,
                           unsigned int PinholeCalib_p_k_num_alloc,
                           const float* const beta,
                           float* out_PinholeCalib_p_kp1,
                           unsigned int out_PinholeCalib_p_kp1_num_alloc,
                           size_t problem_size);

}  // namespace caspar