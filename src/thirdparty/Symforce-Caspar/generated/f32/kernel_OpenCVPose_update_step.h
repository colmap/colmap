#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseUpdateStep(
    float *OpenCVPose_step_k, unsigned int OpenCVPose_step_k_num_alloc,
    float *OpenCVPose_p_kp1, unsigned int OpenCVPose_p_kp1_num_alloc,
    const float *const alpha, float *out_OpenCVPose_step_kp1,
    unsigned int out_OpenCVPose_step_kp1_num_alloc, size_t problem_size);

} // namespace caspar