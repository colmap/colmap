#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialCalibAlphaDenominatorOrBetaNumerator(
    double *SimpleRadialCalib_p_kp1,
    unsigned int SimpleRadialCalib_p_kp1_num_alloc, double *SimpleRadialCalib_w,
    unsigned int SimpleRadialCalib_w_num_alloc,
    double *const SimpleRadialCalib_out, size_t problem_size);

} // namespace caspar