#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPose_retract(
    double* SimpleRadialPose,
    unsigned int SimpleRadialPose_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SimpleRadialPose_retracted,
    unsigned int out_SimpleRadialPose_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar