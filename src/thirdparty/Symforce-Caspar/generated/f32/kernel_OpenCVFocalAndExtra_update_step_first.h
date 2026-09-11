#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraUpdateStepFirst(
    float *OpenCVFocalAndExtra_p_kp1,
    unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc, const float *const alpha,
    float *out_OpenCVFocalAndExtra_step_kp1,
    unsigned int out_OpenCVFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size);

} // namespace caspar