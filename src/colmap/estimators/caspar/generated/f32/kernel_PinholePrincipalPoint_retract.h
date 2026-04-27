#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_retract(
    float* PinholePrincipalPoint,
    unsigned int PinholePrincipalPoint_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_PinholePrincipalPoint_retracted,
    unsigned int out_PinholePrincipalPoint_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar