#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeSplitFixedPoseFixedPrincipalPointFixedPointResJac(
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    SharedIndex* focal_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* pose,
    unsigned int pose_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_focal_njtr,
    unsigned int out_focal_njtr_num_alloc,
    float* const out_focal_precond_diag,
    unsigned int out_focal_precond_diag_num_alloc,
    float* const out_focal_precond_tril,
    unsigned int out_focal_precond_tril_num_alloc,
    size_t problem_size);

}  // namespace caspar