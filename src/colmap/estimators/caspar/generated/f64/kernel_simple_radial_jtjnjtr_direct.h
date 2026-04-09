#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void simple_radial_jtjnjtr_direct(double* pose_njtr,
                                  unsigned int pose_njtr_num_alloc,
                                  SharedIndex* pose_njtr_indices,
                                  double* pose_jac,
                                  unsigned int pose_jac_num_alloc,
                                  double* calib_njtr,
                                  unsigned int calib_njtr_num_alloc,
                                  SharedIndex* calib_njtr_indices,
                                  double* calib_jac,
                                  unsigned int calib_jac_num_alloc,
                                  double* point_njtr,
                                  unsigned int point_njtr_num_alloc,
                                  SharedIndex* point_njtr_indices,
                                  double* point_jac,
                                  unsigned int point_jac_num_alloc,
                                  double* const out_pose_njtr,
                                  unsigned int out_pose_njtr_num_alloc,
                                  double* const out_calib_njtr,
                                  unsigned int out_calib_njtr_num_alloc,
                                  double* const out_point_njtr,
                                  unsigned int out_point_njtr_num_alloc,
                                  size_t problem_size);

}  // namespace caspar