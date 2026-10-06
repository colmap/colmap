#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibUpdateStep(
    float *OpenCVCalib_step_k, unsigned int OpenCVCalib_step_k_num_alloc,
    float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
    const float *const alpha, float *out_OpenCVCalib_step_kp1,
    unsigned int out_OpenCVCalib_step_kp1_num_alloc, size_t problem_size);

} // namespace caspar