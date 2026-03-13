#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_update_r(float* Pose_r_k,
                   unsigned int Pose_r_k_num_alloc,
                   float* Pose_w,
                   unsigned int Pose_w_num_alloc,
                   const float* const negalpha,
                   float* out_Pose_r_kp1,
                   unsigned int out_Pose_r_kp1_num_alloc,
                   float* const out_Pose_r_kp1_norm2_tot,
                   size_t problem_size);

}  // namespace caspar