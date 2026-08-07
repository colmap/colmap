#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraRetract(
    float *OpenCVFocalAndExtra, unsigned int OpenCVFocalAndExtra_num_alloc,
    float *delta, unsigned int delta_num_alloc,
    float *out_OpenCVFocalAndExtra_retracted,
    unsigned int out_OpenCVFocalAndExtra_retracted_num_alloc,
    size_t problem_size);

} // namespace caspar