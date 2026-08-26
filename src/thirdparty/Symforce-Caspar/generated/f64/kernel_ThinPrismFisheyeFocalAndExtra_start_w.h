#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraStartW(
    double* ThinPrismFisheyeFocalAndExtra_precond_diag,
    unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyeFocalAndExtra_p,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_num_alloc,
    double* out_ThinPrismFisheyeFocalAndExtra_w,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar