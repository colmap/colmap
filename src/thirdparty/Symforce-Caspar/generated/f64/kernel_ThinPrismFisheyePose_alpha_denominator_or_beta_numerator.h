#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void ThinPrismFisheyePoseAlphaDenominatorOrBetaNumerator(
    double *ThinPrismFisheyePose_p_kp1,
    unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
    double *ThinPrismFisheyePose_w,
    unsigned int ThinPrismFisheyePose_w_num_alloc,
    double *const ThinPrismFisheyePose_out, size_t problem_size);

} // namespace caspar