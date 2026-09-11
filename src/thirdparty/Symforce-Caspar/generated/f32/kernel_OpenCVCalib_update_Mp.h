#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibUpdateMp(
    float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
    float *OpenCVCalib_Mp, unsigned int OpenCVCalib_Mp_num_alloc,
    const float *const beta, float *out_OpenCVCalib_Mp_kp1,
    unsigned int out_OpenCVCalib_Mp_kp1_num_alloc, float *out_OpenCVCalib_w,
    unsigned int out_OpenCVCalib_w_num_alloc, size_t problem_size);

} // namespace caspar