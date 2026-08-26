#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void ThinPrismFisheyeFocalAndExtraAlphaDenominatorOrBetaNumerator(
    double* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    double* const ThinPrismFisheyeFocalAndExtra_out,
    size_t problem_size);

}  // namespace caspar