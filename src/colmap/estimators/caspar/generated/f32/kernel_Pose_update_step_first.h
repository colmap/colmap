#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_update_step_first(float* Pose_p_kp1,
                            unsigned int Pose_p_kp1_num_alloc,
                            const float* const alpha,
                            float* out_Pose_step_kp1,
                            unsigned int out_Pose_step_kp1_num_alloc,
                            size_t problem_size);

}  // namespace caspar