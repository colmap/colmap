#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibUpdateRFirst(
    double* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    double* ThinPrismFisheyeCalib_w,
    unsigned int ThinPrismFisheyeCalib_w_num_alloc,
    const double* const negalpha,
    double* out_ThinPrismFisheyeCalib_r_kp1,
    unsigned int out_ThinPrismFisheyeCalib_r_kp1_num_alloc,
    double* const out_ThinPrismFisheyeCalib_r_0_norm2_tot,
    double* const out_ThinPrismFisheyeCalib_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar