#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_update_p(
    float* SimpleRadialCalib_z,
    unsigned int SimpleRadialCalib_z_num_alloc,
    float* SimpleRadialCalib_p_k,
    unsigned int SimpleRadialCalib_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialCalib_p_kp1,
    unsigned int out_SimpleRadialCalib_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar