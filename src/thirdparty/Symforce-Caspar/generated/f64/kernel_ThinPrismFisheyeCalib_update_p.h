#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibUpdateP(
    double* ThinPrismFisheyeCalib_z,
    unsigned int ThinPrismFisheyeCalib_z_num_alloc,
    double* ThinPrismFisheyeCalib_p_k,
    unsigned int ThinPrismFisheyeCalib_p_k_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeCalib_p_kp1,
    unsigned int out_ThinPrismFisheyeCalib_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar