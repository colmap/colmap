#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePoseUpdateStep(
    double* ThinPrismFisheyePose_step_k,
    unsigned int ThinPrismFisheyePose_step_k_num_alloc,
    double* ThinPrismFisheyePose_p_kp1,
    unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
    const double* const alpha,
    double* out_ThinPrismFisheyePose_step_kp1,
    unsigned int out_ThinPrismFisheyePose_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar