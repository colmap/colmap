#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_update_p(
    float* PinholeExtraCalib_z,
    unsigned int PinholeExtraCalib_z_num_alloc,
    float* PinholeExtraCalib_p_k,
    unsigned int PinholeExtraCalib_p_k_num_alloc,
    const float* const beta,
    float* out_PinholeExtraCalib_p_kp1,
    unsigned int out_PinholeExtraCalib_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar