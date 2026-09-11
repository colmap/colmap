#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraUpdateP(
    float *OpenCVFocalAndExtra_z, unsigned int OpenCVFocalAndExtra_z_num_alloc,
    float *OpenCVFocalAndExtra_p_k,
    unsigned int OpenCVFocalAndExtra_p_k_num_alloc, const float *const beta,
    float *out_OpenCVFocalAndExtra_p_kp1,
    unsigned int out_OpenCVFocalAndExtra_p_kp1_num_alloc, size_t problem_size);

} // namespace caspar