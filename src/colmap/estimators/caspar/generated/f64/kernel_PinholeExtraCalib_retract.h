#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeExtraCalib_retract(
    double* PinholeExtraCalib,
    unsigned int PinholeExtraCalib_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_PinholeExtraCalib_retracted,
    unsigned int out_PinholeExtraCalib_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar