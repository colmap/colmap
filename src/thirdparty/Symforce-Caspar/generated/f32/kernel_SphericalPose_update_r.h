#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseUpdateR(float* SphericalPose_r_k,
                          unsigned int SphericalPose_r_k_num_alloc,
                          float* SphericalPose_w,
                          unsigned int SphericalPose_w_num_alloc,
                          const float* const negalpha,
                          float* out_SphericalPose_r_kp1,
                          unsigned int out_SphericalPose_r_kp1_num_alloc,
                          float* const out_SphericalPose_r_kp1_norm2_tot,
                          size_t problem_size);

}  // namespace caspar