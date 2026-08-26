#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibUpdateStep(
    double* ThinPrismFisheyeCalib_step_k,
    unsigned int ThinPrismFisheyeCalib_step_k_num_alloc,
    double* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    const double* const alpha,
    double* out_ThinPrismFisheyeCalib_step_kp1,
    unsigned int out_ThinPrismFisheyeCalib_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar