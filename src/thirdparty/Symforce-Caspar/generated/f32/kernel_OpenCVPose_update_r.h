#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseUpdateR(float *OpenCVPose_r_k,
                       unsigned int OpenCVPose_r_k_num_alloc,
                       float *OpenCVPose_w, unsigned int OpenCVPose_w_num_alloc,
                       const float *const negalpha, float *out_OpenCVPose_r_kp1,
                       unsigned int out_OpenCVPose_r_kp1_num_alloc,
                       float *const out_OpenCVPose_r_kp1_norm2_tot,
                       size_t problem_size);

} // namespace caspar