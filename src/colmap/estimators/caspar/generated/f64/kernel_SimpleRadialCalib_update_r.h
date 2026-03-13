#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_update_r(
    double* SimpleRadialCalib_r_k,
    unsigned int SimpleRadialCalib_r_k_num_alloc,
    double* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    const double* const negalpha,
    double* out_SimpleRadialCalib_r_kp1,
    unsigned int out_SimpleRadialCalib_r_kp1_num_alloc,
    double* const out_SimpleRadialCalib_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar