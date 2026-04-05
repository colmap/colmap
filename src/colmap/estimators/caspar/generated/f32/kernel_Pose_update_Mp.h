#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_update_Mp(float* Pose_r_k,
                    unsigned int Pose_r_k_num_alloc,
                    float* Pose_Mp,
                    unsigned int Pose_Mp_num_alloc,
                    const float* const beta,
                    float* out_Pose_Mp_kp1,
                    unsigned int out_Pose_Mp_kp1_num_alloc,
                    float* out_Pose_w,
                    unsigned int out_Pose_w_num_alloc,
                    size_t problem_size);

}  // namespace caspar