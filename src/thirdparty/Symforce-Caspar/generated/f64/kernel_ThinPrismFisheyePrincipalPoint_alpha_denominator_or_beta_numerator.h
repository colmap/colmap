#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointAlphaDenominatorOrBetaNumerator(
    double* ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_w,
    unsigned int ThinPrismFisheyePrincipalPoint_w_num_alloc,
    double* const ThinPrismFisheyePrincipalPoint_out,
    size_t problem_size);

}  // namespace caspar