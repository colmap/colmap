#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_pose_fixed_focal_and_extra_jtjnjtr_direct(
    float* principal_point_njtr,
    unsigned int principal_point_njtr_num_alloc,
    SharedIndex* principal_point_njtr_indices,
    float* principal_point_jac,
    unsigned int principal_point_jac_num_alloc,
    float* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    float* point_jac,
    unsigned int point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar