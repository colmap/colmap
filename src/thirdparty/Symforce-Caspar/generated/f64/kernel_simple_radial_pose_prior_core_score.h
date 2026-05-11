#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialPosePriorCoreScore(double *pose, unsigned int pose_num_alloc,
                                    SharedIndex *pose_indices,
                                    double *prior_position,
                                    unsigned int prior_position_num_alloc,
                                    double *sqrt_info,
                                    unsigned int sqrt_info_num_alloc,
                                    double *const out_rTr, size_t problem_size);

} // namespace caspar