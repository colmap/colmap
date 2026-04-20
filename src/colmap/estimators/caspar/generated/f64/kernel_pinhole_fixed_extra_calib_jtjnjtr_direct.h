#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_extra_calib_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* focal_njtr,
    unsigned int focal_njtr_num_alloc,
    SharedIndex* focal_njtr_indices,
    double* focal_jac,
    unsigned int focal_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar