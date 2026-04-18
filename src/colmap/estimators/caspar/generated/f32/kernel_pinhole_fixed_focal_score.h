#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_focal_score(float* pose,
                               unsigned int pose_num_alloc,
                               SharedIndex* pose_indices,
                               float* extra_calib,
                               unsigned int extra_calib_num_alloc,
                               SharedIndex* extra_calib_indices,
                               float* point,
                               unsigned int point_num_alloc,
                               SharedIndex* point_indices,
                               float* pixel,
                               unsigned int pixel_num_alloc,
                               float* focal,
                               unsigned int focal_num_alloc,
                               float* const out_rTr,
                               size_t problem_size);

}  // namespace caspar