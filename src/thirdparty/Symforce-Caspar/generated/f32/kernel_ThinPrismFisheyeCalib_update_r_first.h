#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibUpdateRFirst(
    float* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    float* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    const float* const negalpha,
    float* out_ThinPrismFisheyeCalib_r_kp1,
    unsigned int out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
    float* const out_ThinPrismFisheyeCalib_r_0_norm2_tot,
    float* const out_ThinPrismFisheyeCalib_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar