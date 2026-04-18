#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialExtraCalib_retract(
    float* SimpleRadialExtraCalib,
    unsigned int SimpleRadialExtraCalib_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SimpleRadialExtraCalib_retracted,
    unsigned int out_SimpleRadialExtraCalib_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar