#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_start_w_contribute(
    double* PinholeFocalAndExtra_precond_diag,
    unsigned int PinholeFocalAndExtra_precond_diag_num_alloc,
    const double* const diag,
    double* PinholeFocalAndExtra_p,
    unsigned int PinholeFocalAndExtra_p_num_alloc,
    double* out_PinholeFocalAndExtra_w,
    unsigned int out_PinholeFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar