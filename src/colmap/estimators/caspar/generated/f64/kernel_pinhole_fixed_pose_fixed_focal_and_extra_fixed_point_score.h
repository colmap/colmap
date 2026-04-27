#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_pose_fixed_focal_and_extra_fixed_point_score(
    double* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* const out_rTr,
    size_t problem_size);

}  // namespace caspar