#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVPrincipalPointRetract(
    float *OpenCVPrincipalPoint, unsigned int OpenCVPrincipalPoint_num_alloc,
    float *delta, unsigned int delta_num_alloc,
    float *out_OpenCVPrincipalPoint_retracted,
    unsigned int out_OpenCVPrincipalPoint_retracted_num_alloc,
    size_t problem_size);

} // namespace caspar