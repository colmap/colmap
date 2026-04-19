#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void PinholeFocalAndExtra_pred_decrease_times_two(
    float* PinholeFocalAndExtra_step,
    unsigned int PinholeFocalAndExtra_step_num_alloc,
    float* PinholeFocalAndExtra_precond_diag,
    unsigned int PinholeFocalAndExtra_precond_diag_num_alloc,
    const float* const diag,
    float* PinholeFocalAndExtra_njtr,
    unsigned int PinholeFocalAndExtra_njtr_num_alloc,
    float* const out_PinholeFocalAndExtra_pred_dec,
    size_t problem_size);

}  // namespace caspar