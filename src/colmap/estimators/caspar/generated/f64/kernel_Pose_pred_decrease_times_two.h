#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_pred_decrease_times_two(double* Pose_step,
                                  unsigned int Pose_step_num_alloc,
                                  double* Pose_precond_diag,
                                  unsigned int Pose_precond_diag_num_alloc,
                                  const double* const diag,
                                  double* Pose_njtr,
                                  unsigned int Pose_njtr_num_alloc,
                                  double* const out_Pose_pred_dec,
                                  size_t problem_size);

}  // namespace caspar