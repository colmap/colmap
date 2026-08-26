#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointAlphaNumeratorDenominator(
    double* ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_r_k,
    unsigned int ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_w,
    unsigned int ThinPrismFisheyePrincipalPoint_w_num_alloc,
    double* const ThinPrismFisheyePrincipalPoint_total_ag,
    double* const ThinPrismFisheyePrincipalPoint_total_ac,
    size_t problem_size);

}  // namespace caspar