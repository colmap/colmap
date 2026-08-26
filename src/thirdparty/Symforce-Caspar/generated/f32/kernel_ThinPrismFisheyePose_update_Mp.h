#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePoseUpdateMp(
    float* ThinPrismFisheyePose_r_k,
    unsigned int ThinPrismFisheyePose_r_k_num_alloc,
    float* ThinPrismFisheyePose_Mp,
    unsigned int ThinPrismFisheyePose_Mp_num_alloc,
    const float* const beta,
    float* out_ThinPrismFisheyePose_Mp_kp1,
    unsigned int out_ThinPrismFisheyePose_Mp_kp1_num_alloc,
    float* out_ThinPrismFisheyePose_w,
    unsigned int out_ThinPrismFisheyePose_w_num_alloc,
    size_t problem_size);

}  // namespace caspar