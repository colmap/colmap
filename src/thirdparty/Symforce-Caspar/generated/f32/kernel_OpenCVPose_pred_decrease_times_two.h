#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPosePredDecreaseTimesTwo(
    float *OpenCVPose_step, unsigned int OpenCVPose_step_num_alloc,
    float *OpenCVPose_precond_diag,
    unsigned int OpenCVPose_precond_diag_num_alloc, const float *const diag,
    float *OpenCVPose_njtr, unsigned int OpenCVPose_njtr_num_alloc,
    float *const out_OpenCVPose_pred_dec, size_t problem_size);

} // namespace caspar