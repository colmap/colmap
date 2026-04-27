#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_alpha_denumerator_or_beta_nummerator(
    double* PinholeFocalAndExtra_p_kp1,
    unsigned int PinholeFocalAndExtra_p_kp1_num_alloc,
    double* PinholeFocalAndExtra_w,
    unsigned int PinholeFocalAndExtra_w_num_alloc,
    double* const PinholeFocalAndExtra_out,
    size_t problem_size);

}  // namespace caspar