#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePrincipalPointStartW(
    double* ThinPrismFisheyePrincipalPoint_precond_diag,
    unsigned int ThinPrismFisheyePrincipalPoint_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyePrincipalPoint_p,
    unsigned int ThinPrismFisheyePrincipalPoint_p_num_alloc,
    double* out_ThinPrismFisheyePrincipalPoint_w,
    unsigned int out_ThinPrismFisheyePrincipalPoint_w_num_alloc,
    size_t problem_size);

}  // namespace caspar