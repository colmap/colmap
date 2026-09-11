#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseStartW(float *OpenCVPose_precond_diag,
                      unsigned int OpenCVPose_precond_diag_num_alloc,
                      const float *const diag, float *OpenCVPose_p,
                      unsigned int OpenCVPose_p_num_alloc,
                      float *out_OpenCVPose_w,
                      unsigned int out_OpenCVPose_w_num_alloc,
                      size_t problem_size);

} // namespace caspar