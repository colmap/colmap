#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_focal_fixed_extra_calib_fixed_point_res_jac_first(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar