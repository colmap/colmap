#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_update_r_first(double* Pose_r_k,
                         unsigned int Pose_r_k_num_alloc,
                         double* Pose_w,
                         unsigned int Pose_w_num_alloc,
                         const double* const negalpha,
                         double* out_Pose_r_kp1,
                         unsigned int out_Pose_r_kp1_num_alloc,
                         double* const out_Pose_r_0_norm2_tot,
                         double* const out_Pose_r_kp1_norm2_tot,
                         size_t problem_size);

}  // namespace caspar