#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointPredDecreaseTimesTwo(
    float* ThinPrismFisheyePrincipalPoint_step,
    unsigned int ThinPrismFisheyePrincipalPoint_step_num_alloc,
    float* ThinPrismFisheyePrincipalPoint_precond_diag,
    unsigned int ThinPrismFisheyePrincipalPoint_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyePrincipalPoint_njtr,
    unsigned int ThinPrismFisheyePrincipalPoint_njtr_num_alloc,
    float* const out_ThinPrismFisheyePrincipalPoint_pred_dec,
    size_t problem_size);

}  // namespace caspar