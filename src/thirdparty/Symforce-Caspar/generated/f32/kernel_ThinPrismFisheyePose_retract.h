#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ThinPrismFisheyePoseRetract(
    float *ThinPrismFisheyePose, unsigned int ThinPrismFisheyePose_num_alloc,
    float *delta, unsigned int delta_num_alloc,
    float *out_ThinPrismFisheyePose_retracted,
    unsigned int out_ThinPrismFisheyePose_retracted_num_alloc,
    size_t problem_size);

} // namespace caspar