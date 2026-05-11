#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void PinholePosePriorCoreResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *prior_position, unsigned int prior_position_num_alloc,
    float *sqrt_info, unsigned int sqrt_info_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size);

} // namespace caspar