#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseUpdateP(float *OpenCVPose_z, unsigned int OpenCVPose_z_num_alloc,
                       float *OpenCVPose_p_k,
                       unsigned int OpenCVPose_p_k_num_alloc,
                       const float *const beta, float *out_OpenCVPose_p_kp1,
                       unsigned int out_OpenCVPose_p_kp1_num_alloc,
                       size_t problem_size);

} // namespace caspar