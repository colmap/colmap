#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseUpdateP(double* SphericalPose_z,
                          unsigned int SphericalPose_z_num_alloc,
                          double* SphericalPose_p_k,
                          unsigned int SphericalPose_p_k_num_alloc,
                          const double* const beta,
                          double* out_SphericalPose_p_kp1,
                          unsigned int out_SphericalPose_p_kp1_num_alloc,
                          size_t problem_size);

}  // namespace caspar