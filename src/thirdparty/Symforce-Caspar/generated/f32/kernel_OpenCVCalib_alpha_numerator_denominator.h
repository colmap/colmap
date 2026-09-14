#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibAlphaNumeratorDenominator(
    float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
    float *OpenCVCalib_r_k, unsigned int OpenCVCalib_r_k_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    float *const OpenCVCalib_total_ag, float *const OpenCVCalib_total_ac,
    size_t problem_size);

} // namespace caspar