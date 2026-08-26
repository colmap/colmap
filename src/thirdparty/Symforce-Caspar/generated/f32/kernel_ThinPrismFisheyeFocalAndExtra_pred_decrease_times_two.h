#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraPredDecreaseTimesTwo(
    float* ThinPrismFisheyeFocalAndExtra_step,
    unsigned int ThinPrismFisheyeFocalAndExtra_step_num_alloc,
    float* ThinPrismFisheyeFocalAndExtra_precond_diag,
    unsigned int ThinPrismFisheyeFocalAndExtra_precond_diag_num_alloc,
    const float* const diag,
    float* ThinPrismFisheyeFocalAndExtra_njtr,
    unsigned int ThinPrismFisheyeFocalAndExtra_njtr_num_alloc,
    float* const out_ThinPrismFisheyeFocalAndExtra_pred_dec,
    size_t problem_size);

}  // namespace caspar