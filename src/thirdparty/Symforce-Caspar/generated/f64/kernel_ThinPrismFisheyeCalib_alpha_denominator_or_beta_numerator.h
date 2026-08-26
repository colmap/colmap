#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibAlphaDenominatorOrBetaNumerator(
    double* ThinPrismFisheyeCalib_p_kp1,
    unsigned int ThinPrismFisheyeCalib_p_kp1_num_alloc,
    double* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    double* const ThinPrismFisheyeCalib_out,
    size_t problem_size);

}  // namespace caspar