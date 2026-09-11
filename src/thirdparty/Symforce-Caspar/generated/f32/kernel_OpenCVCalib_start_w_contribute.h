#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibStartWContribute(
    float *OpenCVCalib_precond_diag,
    unsigned int OpenCVCalib_precond_diag_num_alloc, const float *const diag,
    float *OpenCVCalib_p, unsigned int OpenCVCalib_p_num_alloc,
    float *out_OpenCVCalib_w, unsigned int out_OpenCVCalib_w_num_alloc,
    size_t problem_size);

} // namespace caspar