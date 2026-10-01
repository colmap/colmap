#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ThinPrismFisheyePosePredDecreaseTimesTwo(
    float *ThinPrismFisheyePose_step,
    unsigned int ThinPrismFisheyePose_step_num_alloc,
    float *ThinPrismFisheyePose_precond_diag,
    unsigned int ThinPrismFisheyePose_precond_diag_num_alloc,
    const float *const diag, float *ThinPrismFisheyePose_njtr,
    unsigned int ThinPrismFisheyePose_njtr_num_alloc,
    float *const out_ThinPrismFisheyePose_pred_dec, size_t problem_size);

} // namespace caspar