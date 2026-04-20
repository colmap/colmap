#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void SimpleRadialFocalAndExtra_start_w(
    double* SimpleRadialFocalAndExtra_precond_diag,
    unsigned int SimpleRadialFocalAndExtra_precond_diag_num_alloc,
    const double* const diag,
    double* SimpleRadialFocalAndExtra_p,
    unsigned int SimpleRadialFocalAndExtra_p_num_alloc,
    double* out_SimpleRadialFocalAndExtra_w,
    unsigned int out_SimpleRadialFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar