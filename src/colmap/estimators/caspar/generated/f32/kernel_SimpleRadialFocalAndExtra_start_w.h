#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_start_w(
    float* SimpleRadialFocalAndExtra_precond_diag,
    unsigned int SimpleRadialFocalAndExtra_precond_diag_num_alloc,
    const float* const diag,
    float* SimpleRadialFocalAndExtra_p,
    unsigned int SimpleRadialFocalAndExtra_p_num_alloc,
    float* out_SimpleRadialFocalAndExtra_w,
    unsigned int out_SimpleRadialFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar