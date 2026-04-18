#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_fixed_pose_fixed_extra_calib_fixed_point_res_jac(
    double* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    double* pixel,
    unsigned int pixel_num_alloc,
    double* pose,
    unsigned int pose_num_alloc,
    double* extra_calib,
    unsigned int extra_calib_num_alloc,
    double* point,
    unsigned int point_num_alloc,
    double* out_res,
    unsigned int out_res_num_alloc,
    double* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    double* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    double* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar