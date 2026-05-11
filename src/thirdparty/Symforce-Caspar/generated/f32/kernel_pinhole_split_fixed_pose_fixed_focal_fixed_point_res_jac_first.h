#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void PinholeSplitFixedPoseFixedFocalFixedPointResJacFirst(
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
    float *focal, unsigned int focal_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr,
    float *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size);

} // namespace caspar