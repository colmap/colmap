#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointAlphaNumeratorDenominator(
    float *OpenCVPrincipalPoint_p_kp1,
    unsigned int OpenCVPrincipalPoint_p_kp1_num_alloc,
    float *OpenCVPrincipalPoint_r_k,
    unsigned int OpenCVPrincipalPoint_r_k_num_alloc,
    float *OpenCVPrincipalPoint_w,
    unsigned int OpenCVPrincipalPoint_w_num_alloc,
    float *const OpenCVPrincipalPoint_total_ag,
    float *const OpenCVPrincipalPoint_total_ac, size_t problem_size);

} // namespace caspar