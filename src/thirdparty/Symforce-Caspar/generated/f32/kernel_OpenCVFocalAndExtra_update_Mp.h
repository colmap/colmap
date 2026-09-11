#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraUpdateMp(
    float *OpenCVFocalAndExtra_r_k,
    unsigned int OpenCVFocalAndExtra_r_k_num_alloc,
    float *OpenCVFocalAndExtra_Mp,
    unsigned int OpenCVFocalAndExtra_Mp_num_alloc, const float *const beta,
    float *out_OpenCVFocalAndExtra_Mp_kp1,
    unsigned int out_OpenCVFocalAndExtra_Mp_kp1_num_alloc,
    float *out_OpenCVFocalAndExtra_w,
    unsigned int out_OpenCVFocalAndExtra_w_num_alloc, size_t problem_size);

} // namespace caspar