#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraUpdateRFirst(
    float *OpenCVFocalAndExtra_r_k,
    unsigned int OpenCVFocalAndExtra_r_k_num_alloc,
    float *OpenCVFocalAndExtra_w, unsigned int OpenCVFocalAndExtra_w_num_alloc,
    const float *const negalpha, float *out_OpenCVFocalAndExtra_r_kp1,
    unsigned int out_OpenCVFocalAndExtra_r_kp1_num_alloc,
    float *const out_OpenCVFocalAndExtra_r_0_norm2_tot,
    float *const out_OpenCVFocalAndExtra_r_kp1_norm2_tot, size_t problem_size);

} // namespace caspar