#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointRetract(
    float* ThinPrismFisheyePrincipalPoint,
    unsigned int ThinPrismFisheyePrincipalPoint_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_ThinPrismFisheyePrincipalPoint_retracted,
    unsigned int out_ThinPrismFisheyePrincipalPoint_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar