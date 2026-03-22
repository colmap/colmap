#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_start_w_contribute(double* Pose_precond_diag,
                             unsigned int Pose_precond_diag_num_alloc,
                             const double* const diag,
                             double* Pose_p,
                             unsigned int Pose_p_num_alloc,
                             double* out_Pose_w,
                             unsigned int out_Pose_w_num_alloc,
                             size_t problem_size);

}  // namespace caspar