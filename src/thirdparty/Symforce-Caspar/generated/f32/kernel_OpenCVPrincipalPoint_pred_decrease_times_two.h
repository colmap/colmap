#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointPredDecreaseTimesTwo(
    float *OpenCVPrincipalPoint_step,
    unsigned int OpenCVPrincipalPoint_step_num_alloc,
    float *OpenCVPrincipalPoint_precond_diag,
    unsigned int OpenCVPrincipalPoint_precond_diag_num_alloc,
    const float *const diag, float *OpenCVPrincipalPoint_njtr,
    unsigned int OpenCVPrincipalPoint_njtr_num_alloc,
    float *const out_OpenCVPrincipalPoint_pred_dec, size_t problem_size);

} // namespace caspar