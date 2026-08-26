#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointUpdateMp(
    double* ThinPrismFisheyePrincipalPoint_r_k,
    unsigned int ThinPrismFisheyePrincipalPoint_r_k_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_Mp,
    unsigned int ThinPrismFisheyePrincipalPoint_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyePrincipalPoint_Mp_kp1,
    unsigned int out_ThinPrismFisheyePrincipalPoint_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyePrincipalPoint_w,
    unsigned int out_ThinPrismFisheyePrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar