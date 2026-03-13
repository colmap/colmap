#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_start_w_contribute(float* Pose_precond_diag,
                             unsigned int Pose_precond_diag_num_alloc,
                             const float* const diag,
                             float* Pose_p,
                             unsigned int Pose_p_num_alloc,
                             float* out_Pose_w,
                             unsigned int out_Pose_w_num_alloc,
                             size_t problem_size);

}  // namespace caspar