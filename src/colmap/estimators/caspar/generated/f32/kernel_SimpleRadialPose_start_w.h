#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_start_w(
    float* SimpleRadialPose_precond_diag,
    unsigned int SimpleRadialPose_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialPose_p,
    unsigned int SimpleRadialPose_p_num_alloc,
    float* out_SimpleRadialPose_w,
    unsigned int out_SimpleRadialPose_w_num_alloc,
    size_t problem_size);

}  // namespace caspar