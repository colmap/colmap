#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointUpdateMp(
    float *OpenCVPrincipalPoint_r_k,
    unsigned int OpenCVPrincipalPoint_r_k_num_alloc,
    float *OpenCVPrincipalPoint_Mp,
    unsigned int OpenCVPrincipalPoint_Mp_num_alloc, const float *const beta,
    float *out_OpenCVPrincipalPoint_Mp_kp1,
    unsigned int out_OpenCVPrincipalPoint_Mp_kp1_num_alloc,
    float *out_OpenCVPrincipalPoint_w,
    unsigned int out_OpenCVPrincipalPoint_w_num_alloc, size_t problem_size);

} // namespace caspar