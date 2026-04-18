#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void pinhole_fixed_pose_res_jac(
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* extra_calib,
    unsigned int extra_calib_num_alloc,
    SharedIndex* extra_calib_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_focal_jac,
    unsigned int out_focal_jac_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    float* out_extra_calib_jac,
    unsigned int out_extra_calib_jac_num_alloc,
    float* const out_extra_calib_njtr,
    unsigned int out_extra_calib_njtr_num_alloc,
    float* const out_extra_calib_precond_diag,
    unsigned int out_extra_calib_precond_diag_num_alloc,
    float* const out_extra_calib_precond_tril,
    unsigned int out_extra_calib_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar