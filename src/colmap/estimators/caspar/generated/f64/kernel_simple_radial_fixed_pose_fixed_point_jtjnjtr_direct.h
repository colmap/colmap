#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
    double* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    double* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    double* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    double* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar