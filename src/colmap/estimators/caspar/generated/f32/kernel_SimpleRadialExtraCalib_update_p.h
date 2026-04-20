#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_update_p(
    float* SimpleRadialExtraCalib_z,
    unsigned int SimpleRadialExtraCalib_z_num_alloc,
    float* SimpleRadialExtraCalib_p_k,
    unsigned int SimpleRadialExtraCalib_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialExtraCalib_p_kp1,
    unsigned int out_SimpleRadialExtraCalib_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar