#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraAlphaNumeratorDenominator(
    float *OpenCVFocalAndExtra_p_kp1,
    unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc,
    float *OpenCVFocalAndExtra_r_k,
    unsigned int OpenCVFocalAndExtra_r_k_num_alloc,
    float *OpenCVFocalAndExtra_w, unsigned int OpenCVFocalAndExtra_w_num_alloc,
    float *const OpenCVFocalAndExtra_total_ag,
    float *const OpenCVFocalAndExtra_total_ac, size_t problem_size);

} // namespace caspar