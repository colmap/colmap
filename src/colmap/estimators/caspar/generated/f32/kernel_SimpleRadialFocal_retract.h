#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocal_retract(
    float* SimpleRadialFocal,
    unsigned int SimpleRadialFocal_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SimpleRadialFocal_retracted,
    unsigned int out_SimpleRadialFocal_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar