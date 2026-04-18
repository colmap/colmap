#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_alpha_numerator_denominator(
    float* PinholeFocal_p_kp1,
    unsigned int PinholeFocal_p_kp1_num_alloc,
    float* PinholeFocal_r_k,
    unsigned int PinholeFocal_r_k_num_alloc,
    float* PinholeFocal_w,
    unsigned int PinholeFocal_w_num_alloc,
    float* const PinholeFocal_total_ag,
    float* const PinholeFocal_total_ac,
    size_t problem_size);

}  // namespace caspar