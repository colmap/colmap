#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointUpdateR(
    double* ThinPrismFisheyePrincipalPoint_r_k,
    unsigned int ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_w,
    unsigned int ThinPrismFisheyePrincipalPoint_w_num_alloc,
    const double* const negalpha,
    double* out_ThinPrismFisheyePrincipalPoint_r_kp1,
    unsigned int out_ThinPrismFisheyePrincipalPoint_r_kp1_num_alloc,
    double* const out_ThinPrismFisheyePrincipalPoint_r_kp1_norm2_tot,
    size_t problem_size);

}  // namespace caspar