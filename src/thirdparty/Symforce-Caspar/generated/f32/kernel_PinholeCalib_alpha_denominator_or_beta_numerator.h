#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeCalibAlphaDenominatorOrBetaNumerator(
    float* PinholeCalib_p_kp1,
    unsigned int PinholeCalib_p_kp1_num_alloc,
    float* PinholeCalib_w,
    unsigned int PinholeCalib_w_num_alloc,
    float* const PinholeCalib_out,
    size_t problem_size);

}  // namespace caspar