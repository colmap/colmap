#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionRetract(
    double* SimpleRadialFocalAndDistortion,
    unsigned int SimpleRadialFocalAndDistortion_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_SimpleRadialFocalAndDistortion_retracted,
    unsigned int out_SimpleRadialFocalAndDistortion_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar