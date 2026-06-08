#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PointAlphaDenominatorOrBetaNumerator(double* Point_p_kp1,
                                          unsigned int Point_p_kp1_num_alloc,
                                          double* Point_w,
                                          unsigned int Point_w_num_alloc,
                                          double* const Point_out,
                                          size_t problem_size);

}  // namespace caspar