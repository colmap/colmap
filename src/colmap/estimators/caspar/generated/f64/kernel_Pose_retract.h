#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_retract(double* Pose,
                  unsigned int Pose_num_alloc,
                  double* delta,
                  unsigned int delta_num_alloc,
                  double* out_Pose_retracted,
                  unsigned int out_Pose_retracted_num_alloc,
                  size_t problem_size);

}  // namespace caspar