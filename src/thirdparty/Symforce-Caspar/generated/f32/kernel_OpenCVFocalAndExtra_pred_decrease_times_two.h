#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraPredDecreaseTimesTwo(
    float *OpenCVFocalAndExtra_step,
    unsigned int OpenCVFocalAndExtra_step_num_alloc,
    float *OpenCVFocalAndExtra_precond_diag,
    unsigned int OpenCVFocalAndExtra_precond_diag_num_alloc,
    const float *const diag, float *OpenCVFocalAndExtra_njtr,
    unsigned int OpenCVFocalAndExtra_njtr_num_alloc,
    float *const out_OpenCVFocalAndExtra_pred_dec, size_t problem_size);

} // namespace caspar