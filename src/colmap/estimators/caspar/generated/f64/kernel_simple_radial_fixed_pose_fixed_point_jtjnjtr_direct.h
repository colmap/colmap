#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_pose_fixed_point_jtjnjtr_direct(
    double* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    double* calib_jac,
    unsigned int calib_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    size_t problem_size);

}  // namespace caspar