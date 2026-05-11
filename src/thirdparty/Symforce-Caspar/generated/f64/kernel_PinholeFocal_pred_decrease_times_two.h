#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalPredDecreaseTimesTwo(
    double* PinholeFocal_step,
    unsigned int PinholeFocal_step_num_alloc,
    double* PinholeFocal_precond_diag,
    unsigned int PinholeFocal_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeFocal_njtr,
    unsigned int PinholeFocal_njtr_num_alloc,
    double* const out_PinholeFocal_pred_dec,
    size_t problem_size);

}  // namespace caspar