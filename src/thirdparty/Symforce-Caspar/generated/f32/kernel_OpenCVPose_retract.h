#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPoseRetract(float *OpenCVPose, unsigned int OpenCVPose_num_alloc,
                       float *delta, unsigned int delta_num_alloc,
                       float *out_OpenCVPose_retracted,
                       unsigned int out_OpenCVPose_retracted_num_alloc,
                       size_t problem_size);

} // namespace caspar