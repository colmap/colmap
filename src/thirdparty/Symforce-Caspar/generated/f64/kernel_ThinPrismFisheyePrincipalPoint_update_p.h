#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointUpdateP(
    double* ThinPrismFisheyePrincipalPoint_z,
    unsigned int ThinPrismFisheyePrincipalPoint_z_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_p_k,
    unsigned int ThinPrismFisheyePrincipalPoint_p_k_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int out_ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar