#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibUpdateR(
    float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    const float *const negalpha, float *out_OpenCVCalib_r_kp1,
    unsigned int out_OpenCVCalib_r_kp1_num_alloc,
    float *const out_OpenCVCalib_r_kp1_norm2_tot, size_t problem_size);

} // namespace caspar