#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointUpdateStep(
    double* ThinPrismFisheyePrincipalPoint_step_k,
    unsigned int ThinPrismFisheyePrincipalPoint_step_k_num_alloc,
    double* ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    const double* const alpha,
    double* out_ThinPrismFisheyePrincipalPoint_step_kp1,
    unsigned int out_ThinPrismFisheyePrincipalPoint_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar