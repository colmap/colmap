#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionAlphaDenominatorOrBetaNumerator(
    double* SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    double* SimpleRadialFocalAndDistortion_w,
    unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
    double* const SimpleRadialFocalAndDistortion_out,
    size_t problem_size);

}  // namespace caspar