#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibPredDecreaseTimesTwo(
    float *OpenCVCalib_step, unsigned int OpenCVCalib_step_num_alloc,
    float *OpenCVCalib_precond_diag,
    unsigned int OpenCVCalib_precond_diag_num_alloc, const float *const diag,
    float *OpenCVCalib_njtr, unsigned int OpenCVCalib_njtr_num_alloc,
    float *const out_OpenCVCalib_pred_dec, size_t problem_size);

} // namespace caspar