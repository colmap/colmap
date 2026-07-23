#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseAlphaDenominatorOrBetaNumerator(
    double* SphericalPose_p_kp1,
    unsigned int SphericalPose_p_kp1_num_alloc,
    double* SphericalPose_w,
    unsigned int SphericalPose_w_num_alloc,
    double* const SphericalPose_out,
    size_t problem_size);

}  // namespace caspar