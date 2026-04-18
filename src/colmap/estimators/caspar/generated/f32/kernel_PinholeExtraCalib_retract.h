#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_retract(
    float* PinholeExtraCalib,
    unsigned int PinholeExtraCalib_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_PinholeExtraCalib_retracted,
    unsigned int out_PinholeExtraCalib_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar