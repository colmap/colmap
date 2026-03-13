#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_pred_decrease_times_two(float* Pose_step,
                                  unsigned int Pose_step_num_alloc,
                                  float* Pose_precond_diag,
                                  unsigned int Pose_precond_diag_num_alloc,
                                  const float* const diag,
                                  float* Pose_njtr,
                                  unsigned int Pose_njtr_num_alloc,
                                  float* const out_Pose_pred_dec,
                                  size_t problem_size);

}  // namespace caspar