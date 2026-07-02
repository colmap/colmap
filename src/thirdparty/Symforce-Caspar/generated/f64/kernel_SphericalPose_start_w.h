#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SphericalPoseStartW(double* SphericalPose_precond_diag,
                         unsigned int SphericalPose_precond_diag_num_alloc,
                         const double* const diag,
                         double* SphericalPose_p,
                         unsigned int SphericalPose_p_num_alloc,
                         double* out_SphericalPose_w,
                         unsigned int out_SphericalPose_w_num_alloc,
                         size_t problem_size);

}  // namespace caspar