#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_start_w_contribute(
    float* PinholeFocal_precond_diag,
    unsigned int PinholeFocal_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeFocal_p,
    unsigned int PinholeFocal_p_num_alloc,
    float* out_PinholeFocal_w,
    unsigned int out_PinholeFocal_w_num_alloc,
    size_t problem_size);

}  // namespace caspar