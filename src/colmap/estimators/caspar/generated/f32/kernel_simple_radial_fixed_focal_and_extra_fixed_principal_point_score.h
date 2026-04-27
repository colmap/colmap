#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_focal_and_extra_fixed_principal_point_score(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* const out_rTr,
    size_t problem_size);

}  // namespace caspar