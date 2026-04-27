#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_merged_score(float* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          float* calib,
                          unsigned int calib_num_alloc,
                          SharedIndex* calib_indices,
                          float* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          float* pixel,
                          unsigned int pixel_num_alloc,
                          float* const out_rTr,
                          size_t problem_size);

}  // namespace caspar