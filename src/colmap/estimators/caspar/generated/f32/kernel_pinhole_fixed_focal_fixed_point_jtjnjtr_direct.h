#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_focal_fixed_point_jtjnjtr_direct(
    float* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    float* pose_jac,
    unsigned int pose_jac_num_alloc,
    float* extra_calib_njtr,
    unsigned int extra_calib_njtr_num_alloc,
    SharedIndex* extra_calib_njtr_indices,
    float* extra_calib_jac,
    unsigned int extra_calib_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar