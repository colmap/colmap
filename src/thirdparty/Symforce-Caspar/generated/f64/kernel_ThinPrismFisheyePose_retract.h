#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ThinPrismFisheyePoseRetract(
    double *ThinPrismFisheyePose, unsigned int ThinPrismFisheyePose_num_alloc,
    double *delta, unsigned int delta_num_alloc,
    double *out_ThinPrismFisheyePose_retracted,
    unsigned int out_ThinPrismFisheyePose_retracted_num_alloc,
    size_t problem_size);

} // namespace caspar