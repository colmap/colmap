#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void OpenCVFocalAndExtraAlphaDenominatorOrBetaNumerator(
    float *OpenCVFocalAndExtra_p_kp1,
    unsigned int OpenCVFocalAndExtra_p_kp1_num_alloc,
    float *OpenCVFocalAndExtra_w, unsigned int OpenCVFocalAndExtra_w_num_alloc,
    float *const OpenCVFocalAndExtra_out, size_t problem_size);

} // namespace caspar