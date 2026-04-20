#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_extra_calib_fixed_point_score(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* const out_rTr,
    size_t problem_size);

}  // namespace caspar