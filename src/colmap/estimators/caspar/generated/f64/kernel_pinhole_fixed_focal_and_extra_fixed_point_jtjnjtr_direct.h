#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_focal_and_extra_fixed_point_jtjnjtr_direct(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    double* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar