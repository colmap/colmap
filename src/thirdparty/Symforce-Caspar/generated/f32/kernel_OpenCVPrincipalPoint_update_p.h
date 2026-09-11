#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointUpdateP(
    float *OpenCVPrincipalPoint_z,
    unsigned int OpenCVPrincipalPoint_z_num_alloc,
    float *OpenCVPrincipalPoint_p_k,
    unsigned int OpenCVPrincipalPoint_p_k_num_alloc, const float *const beta,
    float *out_OpenCVPrincipalPoint_p_kp1,
    unsigned int out_OpenCVPrincipalPoint_p_kp1_num_alloc, size_t problem_size);

} // namespace caspar