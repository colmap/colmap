#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_retract(float* Pose,
                  unsigned int Pose_num_alloc,
                  float* delta,
                  unsigned int delta_num_alloc,
                  float* out_Pose_retracted,
                  unsigned int out_Pose_retracted_num_alloc,
                  size_t problem_size);

}  // namespace caspar