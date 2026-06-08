#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePoseAlphaDenominatorOrBetaNumerator(
    float* PinholePose_p_kp1,
    unsigned int PinholePose_p_kp1_num_alloc,
    float* PinholePose_w,
    unsigned int PinholePose_w_num_alloc,
    float* const PinholePose_out,
    size_t problem_size);

}  // namespace caspar