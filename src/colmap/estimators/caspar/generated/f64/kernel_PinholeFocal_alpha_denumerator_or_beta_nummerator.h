#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_alpha_denumerator_or_beta_nummerator(
    double* PinholeFocal_p_kp1,
    unsigned int PinholeFocal_p_kp1_num_alloc,
    double* PinholeFocal_w,
    unsigned int PinholeFocal_w_num_alloc,
    double* const PinholeFocal_out,
    size_t problem_size);

}  // namespace caspar