#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointStartW(
    float *OpenCVPrincipalPoint_precond_diag,
    unsigned int OpenCVPrincipalPoint_precond_diag_num_alloc,
    const float *const diag, float *OpenCVPrincipalPoint_p,
    unsigned int OpenCVPrincipalPoint_p_num_alloc,
    float *out_OpenCVPrincipalPoint_w,
    unsigned int out_OpenCVPrincipalPoint_w_num_alloc, size_t problem_size);

} // namespace caspar