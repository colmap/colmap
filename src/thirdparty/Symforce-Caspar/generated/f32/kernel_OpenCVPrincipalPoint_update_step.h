#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointUpdateStep(
    float *OpenCVPrincipalPoint_step_k,
    unsigned int OpenCVPrincipalPoint_step_k_num_alloc,
    float *OpenCVPrincipalPoint_p_kp1,
    unsigned int OpenCVPrincipalPoint_p_kp1_num_alloc, const float *const alpha,
    float *out_OpenCVPrincipalPoint_step_kp1,
    unsigned int out_OpenCVPrincipalPoint_step_kp1_num_alloc,
    size_t problem_size);

} // namespace caspar