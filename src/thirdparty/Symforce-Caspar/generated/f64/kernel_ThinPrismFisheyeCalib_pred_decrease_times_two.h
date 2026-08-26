#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibPredDecreaseTimesTwo(
    double* ThinPrismFisheyeCalib_step,
    unsigned int ThinPrismFisheyeCalib_step_num_alloc,
    double* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyeCalib_njtr,
    unsigned int ThinPrismFisheyeCalib_njtr_num_alloc,
    double* const out_ThinPrismFisheyeCalib_pred_dec,
    size_t problem_size);

}  // namespace caspar