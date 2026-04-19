#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialPrincipalPoint_retract(
    float* SimpleRadialPrincipalPoint,
    unsigned int SimpleRadialPrincipalPoint_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SimpleRadialPrincipalPoint_retracted,
    unsigned int out_SimpleRadialPrincipalPoint_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar