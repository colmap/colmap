#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseAlphaDenominatorOrBetaNumerator(
    float *OpenCVPose_p_kp1, unsigned int OpenCVPose_p_kp1_num_alloc,
    float *OpenCVPose_w, unsigned int OpenCVPose_w_num_alloc,
    float *const OpenCVPose_out, size_t problem_size);

} // namespace caspar