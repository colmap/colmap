#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_update_Mp(
    double* SimpleRadialFocal_r_k,
    unsigned int SimpleRadialFocal_r_k_num_alloc,
    double* SimpleRadialFocal_Mp,
    unsigned int SimpleRadialFocal_Mp_num_alloc,
    const double* const beta,
    double* out_SimpleRadialFocal_Mp_kp1,
    unsigned int out_SimpleRadialFocal_Mp_kp1_num_alloc,
    double* out_SimpleRadialFocal_w,
    unsigned int out_SimpleRadialFocal_w_num_alloc,
    size_t problem_size);

}  // namespace caspar