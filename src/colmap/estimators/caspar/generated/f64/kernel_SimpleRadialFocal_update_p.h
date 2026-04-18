#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_update_p(
    double* SimpleRadialFocal_z,
    unsigned int SimpleRadialFocal_z_num_alloc,
    double* SimpleRadialFocal_p_k,
    unsigned int SimpleRadialFocal_p_k_num_alloc,
    const double* const beta,
    double* out_SimpleRadialFocal_p_kp1,
    unsigned int out_SimpleRadialFocal_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar