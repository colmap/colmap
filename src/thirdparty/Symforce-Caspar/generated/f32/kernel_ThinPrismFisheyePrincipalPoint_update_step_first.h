#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointUpdateStepFirst(
    float* ThinPrismFisheyePrincipalPoint_p_kp1,
    unsigned int ThinPrismFisheyePrincipalPoint_p_kp1_num_alloc,
    const float* const alpha,
    float* out_ThinPrismFisheyePrincipalPoint_step_kp1,
    unsigned int out_ThinPrismFisheyePrincipalPoint_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar