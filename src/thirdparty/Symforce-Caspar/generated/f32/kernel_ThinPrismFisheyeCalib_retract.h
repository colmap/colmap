#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeCalibRetract(
    float* ThinPrismFisheyeCalib,
    unsigned int ThinPrismFisheyeCalib_num_alloc,
    float* delta,
    unsigned int delta_num_alloc,
    float* out_ThinPrismFisheyeCalib_retracted,
    unsigned int out_ThinPrismFisheyeCalib_retracted_num_alloc,
    size_t problem_size);

}  // namespace caspar