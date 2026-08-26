#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePoseUpdateP(
    double* ThinPrismFisheyePose_z,
    unsigned int ThinPrismFisheyePose_z_num_alloc,
    double* ThinPrismFisheyePose_p_k,
    unsigned int ThinPrismFisheyePose_p_k_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyePose_p_kp1,
    unsigned int out_ThinPrismFisheyePose_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar