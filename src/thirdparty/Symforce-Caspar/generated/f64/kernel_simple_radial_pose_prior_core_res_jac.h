#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void SimpleRadialPosePriorCoreResJac(
    double *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    double *prior_position, unsigned int prior_position_num_alloc,
    double *sqrt_info, unsigned int sqrt_info_num_alloc, double *out_res,
    unsigned int out_res_num_alloc, double *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, double *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    double *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size);

} // namespace caspar