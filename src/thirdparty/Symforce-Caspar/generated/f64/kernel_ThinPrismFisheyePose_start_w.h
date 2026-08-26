#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePoseStartW(
    double* ThinPrismFisheyePose_precond_diag,
    unsigned int ThinPrismFisheyePose_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyePose_p,
    unsigned int ThinPrismFisheyePose_p_num_alloc,
    double* out_ThinPrismFisheyePose_w,
    unsigned int out_ThinPrismFisheyePose_w_num_alloc,
    size_t problem_size);

}  // namespace caspar