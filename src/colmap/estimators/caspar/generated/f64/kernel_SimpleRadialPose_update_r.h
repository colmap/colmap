#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_update_r(
    double* SimpleRadialPose_r_k,
    unsigned int SimpleRadialPose_r_k_num_alloc,
    double* SimpleRadialPose_w,
    unsigned int SimpleRadialPose_w_num_alloc,
    const double* const negalpha,
    double* out_SimpleRadialPose_r_kp1,
    unsigned int out_SimpleRadialPose_r_kp1_num_alloc,
    double* const out_SimpleRadialPose_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar