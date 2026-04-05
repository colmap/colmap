#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void Pose_alpha_numerator_denominator(double* Pose_p_kp1,
                                      unsigned int Pose_p_kp1_num_alloc,
                                      double* Pose_r_k,
                                      unsigned int Pose_r_k_num_alloc,
                                      double* Pose_w,
                                      unsigned int Pose_w_num_alloc,
                                      double* const Pose_total_ag,
                                      double* const Pose_total_ac,
                                      size_t problem_size);

}  // namespace caspar