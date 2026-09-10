#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ThinPrismFisheyePoseUpdateP(
    float *ThinPrismFisheyePose_z,
    unsigned int ThinPrismFisheyePose_z_num_alloc,
    float *ThinPrismFisheyePose_p_k,
    unsigned int ThinPrismFisheyePose_p_k_num_alloc, const float *const beta,
    float *out_ThinPrismFisheyePose_p_kp1,
    unsigned int out_ThinPrismFisheyePose_p_kp1_num_alloc, size_t problem_size);

} // namespace caspar