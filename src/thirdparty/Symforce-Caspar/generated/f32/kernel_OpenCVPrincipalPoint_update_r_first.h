#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointUpdateRFirst(
    float *OpenCVPrincipalPoint_r_k,
    unsigned int OpenCVPrincipalPoint_r_k_num_alloc,
    float *OpenCVPrincipalPoint_w,
    unsigned int OpenCVPrincipalPoint_w_num_alloc, const float *const negalpha,
    float *out_OpenCVPrincipalPoint_r_kp1,
    unsigned int out_OpenCVPrincipalPoint_r_kp1_num_alloc,
    float *const out_OpenCVPrincipalPoint_r_0_norm2_tot,
    float *const out_OpenCVPrincipalPoint_r_kp1_norm2_tot, size_t problem_size);

} // namespace caspar