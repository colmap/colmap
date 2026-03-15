#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_pose_fixed_point_score(float* calib,
                                          unsigned int calib_num_alloc,
                                          SharedIndex* calib_indices,
                                          float* pixel,
                                          unsigned int pixel_num_alloc,
                                          float* pose,
                                          unsigned int pose_num_alloc,
                                          float* point,
                                          unsigned int point_num_alloc,
                                          float* const out_rTr,
                                          size_t problem_size);

}  // namespace caspar