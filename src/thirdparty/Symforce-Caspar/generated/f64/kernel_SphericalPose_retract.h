#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseRetract(double* SphericalPose,
                          unsigned int SphericalPose_num_alloc,
                          double* delta,
                          unsigned int delta_num_alloc,
                          double* out_SphericalPose_retracted,
                          unsigned int out_SphericalPose_retracted_num_alloc,
                          size_t problem_size);

}  // namespace caspar