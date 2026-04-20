#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocal_pred_decrease_times_two(
    float* PinholeFocal_step,
    unsigned int PinholeFocal_step_num_alloc,
    float* PinholeFocal_precond_diag,
    unsigned int PinholeFocal_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeFocal_njtr,
    unsigned int PinholeFocal_njtr_num_alloc,
    float* const out_PinholeFocal_pred_dec,
    size_t problem_size);

}  // namespace caspar