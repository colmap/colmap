#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPosePredDecreaseTimesTwo(
    double* SphericalPose_step,
    unsigned int SphericalPose_step_num_alloc,
    double* SphericalPose_precond_diag,
    unsigned int SphericalPose_precond_diag_num_alloc,
    const double* const diag,
    double* SphericalPose_njtr,
    unsigned int SphericalPose_njtr_num_alloc,
    double* const out_SphericalPose_pred_dec,
    size_t problem_size);

}  // namespace caspar