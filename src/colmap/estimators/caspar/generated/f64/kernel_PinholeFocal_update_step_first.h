#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_update_step_first(
    double* PinholeFocal_p_kp1,
    unsigned int PinholeFocal_p_kp1_num_alloc,
    const double* const alpha,
    double* out_PinholeFocal_step_kp1,
    unsigned int out_PinholeFocal_step_kp1_num_alloc,
    size_t problem_size);

}  // namespace caspar