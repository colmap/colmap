#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalib_retract(
    double* SimpleRadialCalib,
    unsigned int SimpleRadialCalib_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SimpleRadialCalib_retracted,
    unsigned int out_SimpleRadialCalib_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar