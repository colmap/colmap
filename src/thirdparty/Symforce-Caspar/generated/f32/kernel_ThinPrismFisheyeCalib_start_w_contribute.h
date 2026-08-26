#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibStartWContribute(
    float* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeCalib_p,
    unsigned int ThinPrismFisheyeCalib_p_num_alloc,
    float* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar