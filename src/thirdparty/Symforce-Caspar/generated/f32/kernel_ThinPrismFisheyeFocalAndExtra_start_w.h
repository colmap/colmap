#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraStartW(
    float* ThinPrismFisheyeFocalAndExtra_precond_diag,
    unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeFocalAndExtra_p,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_num_alloc,
    float* out_ThinPrismFisheyeFocalAndExtra_w,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    size_t problem_size);

}  // namespace caspar