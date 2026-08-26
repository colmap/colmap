#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyePoseUpdateMp(
    double* ThinPrismFisheyePose_r_k,
    unsigned int ThinPrismFisheyePose_r_k_num_alloc,
    double* ThinPrismFisheyePose_Mp,
    unsigned int ThinPrismFisheyePose_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyePose_Mp_kp1,
    unsigned int out_ThinPrismFisheyePose_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyePose_w,
    unsigned int out_ThinPrismFisheyePose_w_num_alloc,
    size_t problem_size);

}  // namespace caspar