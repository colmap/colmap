#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibUpdateMp(
    double* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    double* ThinPrismFisheyeCalib_Mp,
    unsigned int ThinPrismFisheyeCalib_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeCalib_Mp_kp1,
    unsigned int out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size);

}  // namespace caspar