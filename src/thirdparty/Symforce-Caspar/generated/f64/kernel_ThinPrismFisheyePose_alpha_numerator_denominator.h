#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePoseAlphaNumeratorDenominator(
    double* ThinPrismFisheyePose_p_kp1,
    unsigned int ThinPrismFisheyePose_p_kp1_num_alloc,
    double* ThinPrismFisheyePose_r_k,
    unsigned int ThinPrismFisheyePose_r_k_num_alloc,
    double* ThinPrismFisheyePose_w,
    unsigned int ThinPrismFisheyePose_w_num_alloc,
    double* const ThinPrismFisheyePose_total_ag,
    double* const ThinPrismFisheyePose_total_ac,
    size_t problem_size);

}  // namespace caspar