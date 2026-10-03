#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointAlphaDenominatorOrBetaNumerator(
    float *OpenCVPrincipalPoint_p_kp1,
    unsigned int OpenCVPrincipalPoint_p_kp1_num_alloc,
    float *OpenCVPrincipalPoint_w,
    unsigned int OpenCVPrincipalPoint_w_num_alloc,
    float *const OpenCVPrincipalPoint_out, size_t problem_size);

} // namespace caspar