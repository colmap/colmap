#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_update_p(float* Pose_z,
                   unsigned int Pose_z_num_alloc,
                   float* Pose_p_k,
                   unsigned int Pose_p_k_num_alloc,
                   const float* const beta,
                   float* out_Pose_p_kp1,
                   unsigned int out_Pose_p_kp1_num_alloc,
                   size_t problem_size);

}  // namespace caspar