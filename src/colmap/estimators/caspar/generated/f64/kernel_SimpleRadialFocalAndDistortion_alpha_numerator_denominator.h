#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndDistortionAlphaNumeratorDenominator(
    double* SimpleRadialFocalAndDistortion_p_kp1,
    unsigned int SimpleRadialFocalAndDistortion_p_kp1_num_alloc,
    double* SimpleRadialFocalAndDistortion_r_k,
    unsigned int SimpleRadialFocalAndDistortion_r_k_num_alloc,
    double* SimpleRadialFocalAndDistortion_w,
    unsigned int SimpleRadialFocalAndDistortion_w_num_alloc,
    double* const SimpleRadialFocalAndDistortion_total_ag,
    double* const SimpleRadialFocalAndDistortion_total_ac,
    size_t problem_size);

}  // namespace caspar