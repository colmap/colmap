#pragma once

#include <cuda_runtime.h>

#include "shared_indices.h"

namespace caspar {

void PinholeFocalStartWContribute(
    double *PinholeFocal_precond_diag,
    unsigned int PinholeFocal_precond_diag_num_alloc, const double *const diag,
    double *PinholeFocal_p, unsigned int PinholeFocal_p_num_alloc,
    double *out_PinholeFocal_w, unsigned int out_PinholeFocal_w_num_alloc,
    size_t problem_size);

} // namespace caspar