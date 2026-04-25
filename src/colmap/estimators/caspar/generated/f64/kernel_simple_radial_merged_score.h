#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_merged_score(double* pose,
                                unsigned int pose_num_alloc,
                                SharedIndex* pose_indices,
                                double* calib,
                                unsigned int calib_num_alloc,
                                SharedIndex* calib_indices,
                                double* point,
                                unsigned int point_num_alloc,
                                SharedIndex* point_indices,
                                double* pixel,
                                unsigned int pixel_num_alloc,
                                double* const out_rTr,
                                size_t problem_size);

}  // namespace caspar