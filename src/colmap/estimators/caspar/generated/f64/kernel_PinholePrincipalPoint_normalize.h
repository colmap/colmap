#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholePrincipalPoint_normalize(double* precond_diag,
                                     unsigned int precond_diag_num_alloc,
                                     double* precond_tril,
                                     unsigned int precond_tril_num_alloc,
                                     double* njtr,
                                     unsigned int njtr_num_alloc,
                                     const double* const diag,
                                     double* out_normalized,
                                     unsigned int out_normalized_num_alloc,
                                     size_t problem_size);

}  // namespace caspar