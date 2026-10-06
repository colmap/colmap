#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseUpdateMp(
    float *OpenCVPose_r_k, unsigned int OpenCVPose_r_k_num_alloc,
    float *OpenCVPose_Mp, unsigned int OpenCVPose_Mp_num_alloc,
    const float *const beta, float *out_OpenCVPose_Mp_kp1,
    unsigned int out_OpenCVPose_Mp_kp1_num_alloc, float *out_OpenCVPose_w,
    unsigned int out_OpenCVPose_w_num_alloc, size_t problem_size);

} // namespace caspar