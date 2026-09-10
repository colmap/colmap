#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ThinPrismFisheyePoseUpdateRFirst(
    double *ThinPrismFisheyePose_r_k,
    unsigned int ThinPrismFisheyePose_r_k_num_alloc,
    double *ThinPrismFisheyePose_w,
    unsigned int ThinPrismFisheyePose_w_num_alloc, const double *const negalpha,
    double *out_ThinPrismFisheyePose_r_kp1,
    unsigned int out_ThinPrismFisheyePose_r_kp1_num_alloc,
    double *const out_ThinPrismFisheyePose_r_0_norm2_tot,
    double *const out_ThinPrismFisheyePose_r_kp1_norm2_tot,
    size_t problem_size);

} // namespace caspar