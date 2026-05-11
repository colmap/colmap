#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionRetract(
    float* SimpleRadialFocalAndDistortion,
    unsigned int SimpleRadialFocalAndDistortion_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_SimpleRadialFocalAndDistortion_retracted,
    unsigned int out_SimpleRadialFocalAndDistortion_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar