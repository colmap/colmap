#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_update_p(
    float* SimpleRadialFocal_z,
    unsigned int SimpleRadialFocal_z_num_alloc,
    float* SimpleRadialFocal_p_k,
    unsigned int SimpleRadialFocal_p_k_num_alloc,
    const float* const beta,
    float* out_SimpleRadialFocal_p_kp1,
    unsigned int out_SimpleRadialFocal_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar