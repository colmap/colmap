#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibPredDecreaseTimesTwo(
    float* ThinPrismFisheyeCalib_step,
    unsigned int ThinPrismFisheyeCalib_step_num_alloc,
    float* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeCalib_njtr,
    unsigned int ThinPrismFisheyeCalib_njtr_num_alloc,
    float* const out_ThinPrismFisheyeCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar