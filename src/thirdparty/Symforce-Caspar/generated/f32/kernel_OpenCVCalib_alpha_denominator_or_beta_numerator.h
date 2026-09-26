#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibAlphaDenominatorOrBetaNumerator(
    float *OpenCVCalib_p_kp1, unsigned int OpenCVCalib_p_kp1_num_alloc,
    float *OpenCVCalib_w, unsigned int OpenCVCalib_w_num_alloc,
    float *const OpenCVCalib_out, size_t problem_size);

} // namespace caspar