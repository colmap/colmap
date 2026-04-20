#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_pose_fixed_extra_calib_fixed_point_jtjnjtr_direct(
    float* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    float* focal_jac,
    unsigned int focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar