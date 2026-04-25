#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_update_step_first(
    double* SimpleRadialPose_p_kp1,
    unsigned int SimpleRadialPose_p_kp1_num_alloc,
    const double* const alpha,
    double* out_SimpleRadialPose_step_kp1,
    unsigned int out_SimpleRadialPose_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar