#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialPosePriorCoreScore(float *pose, unsigned int pose_num_alloc,
                                    SharedIndex *pose_indices,
                                    float *prior_position,
                                    unsigned int prior_position_num_alloc,
                                    float *sqrt_info,
                                    unsigned int sqrt_info_num_alloc,
                                    float *const out_rTr, size_t problem_size);

} // namespace caspar