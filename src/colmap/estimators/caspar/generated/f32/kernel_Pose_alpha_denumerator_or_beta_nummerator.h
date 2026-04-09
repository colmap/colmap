#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_alpha_denumerator_or_beta_nummerator(
    float* Pose_p_kp1,
    unsigned int Pose_p_kp1_num_alloc,
    float* Pose_w,
    unsigned int Pose_w_num_alloc,
    float* const Pose_out,
    size_t problem_size);

}  // namespace caspar