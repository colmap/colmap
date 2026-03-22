#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_calib_jtjnjtr_direct(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar