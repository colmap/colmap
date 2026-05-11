#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialCalibAlphaDenominatorOrBetaNumerator(
    float* SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc,
    float* SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    float* const SimpleRadialCalib_out,
    size_t problem_size);

}  // namespace caspar