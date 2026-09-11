#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraStartWContribute(
    float *OpenCVFocalAndExtra_precond_diag,
    unsigned int OpenCVFocalAndExtra_precond_diag_num_alloc,
    const float *const diag, float *OpenCVFocalAndExtra_p,
    unsigned int OpenCVFocalAndExtra_p_num_alloc,
    float *out_OpenCVFocalAndExtra_w,
    unsigned int out_OpenCVFocalAndExtra_w_num_alloc, size_t problem_size);

} // namespace caspar