#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseRetract(float* SphericalPose,
                          unsigned int SphericalPose_num_alloc,
                          float* delta,
                          unsigned int delta_num_alloc,
                          float* out_SphericalPose_retracted,
                          unsigned int out_SphericalPose_retracted_num_alloc,
                          size_t problem_size);

}  // namespace caspar