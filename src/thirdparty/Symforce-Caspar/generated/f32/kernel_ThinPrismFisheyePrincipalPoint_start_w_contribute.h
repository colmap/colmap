#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointStartWContribute(
    float* ThinPrismFisheyePrincipalPoint_precond_diag,
    unsigned int ThinPrismFisheyePrincipalPoint_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyePrincipalPoint_p,
    unsigned int ThinPrismFisheyePrincipalPoint_p_num_alloc,
    float* out_ThinPrismFisheyePrincipalPoint_w,
    unsigned int out_ThinPrismFisheyePrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar