#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPointAlphaDenominatorOrBetaNumerator(
    float* PinholePrincipalPoint_p_kp1,
    unsigned int PinholePrincipalPoint_p_kp1_num_alloc,
    float* PinholePrincipalPoint_w,
    unsigned int PinholePrincipalPoint_w_num_alloc,
    float* const PinholePrincipalPoint_out,
    size_t problem_size);

}  // namespace caspar