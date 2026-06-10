#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseUpdateMp(float* SphericalPose_r_k,
                           unsigned int SphericalPose_r_k_num_alloc,
                           float* SphericalPose_Mp,
                           unsigned int SphericalPose_Mp_num_alloc,
                           const float* const beta,
                           float* out_SphericalPose_Mp_kp1,
                           unsigned int out_SphericalPose_Mp_kp1_num_alloc,
                           float* out_SphericalPose_w,
                           unsigned int out_SphericalPose_w_num_alloc,
                           size_t problem_size);

}  // namespace caspar