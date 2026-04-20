#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_point_fixed_calib_res_jac(
    double* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* calib,
    unsigned int calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar