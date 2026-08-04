#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVCalibRetract(float *OpenCVCalib, unsigned int OpenCVCalib_num_alloc,
                        float *delta, unsigned int delta_num_alloc,
                        float *out_OpenCVCalib_retracted,
                        unsigned int out_OpenCVCalib_retracted_num_alloc,
                        size_t problem_size);

} // namespace caspar