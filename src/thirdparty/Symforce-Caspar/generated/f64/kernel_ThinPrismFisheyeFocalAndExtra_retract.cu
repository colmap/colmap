#include "kernel_ThinPrismFisheyeFocalAndExtra_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraRetractKernel(
        double* ThinPrismFisheyeFocalAndExtra,
        unsigned int ThinPrismFisheyeFocalAndExtra_num_alloc,
        double* delta,
        unsigned int delta_num_alloc,
        double* out_ThinPrismFisheyeFocalAndExtra_retracted,
        unsigned int out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra,
        0 * ThinPrismFisheyeFocalAndExtra_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_retracted,
        0 * out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra,
        2 * ThinPrismFisheyeFocalAndExtra_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        delta, 2 * delta_num_alloc, global_thread_idx, r1, r0);
    r1 = r3 + r1;
    r0 = r2 + r0;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_retracted,
        2 * out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra,
        4 * ThinPrismFisheyeFocalAndExtra_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_retracted,
        4 * out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra,
        6 * ThinPrismFisheyeFocalAndExtra_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        delta, 6 * delta_num_alloc, global_thread_idx, r1, r0);
    r1 = r3 + r1;
    r0 = r2 + r0;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_retracted,
        6 * out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra,
        8 * ThinPrismFisheyeFocalAndExtra_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        delta, 8 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_retracted,
        8 * out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
}

void ThinPrismFisheyeFocalAndExtraRetract(
    double* ThinPrismFisheyeFocalAndExtra,
    unsigned int ThinPrismFisheyeFocalAndExtra_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_ThinPrismFisheyeFocalAndExtra_retracted,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraRetractKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeFocalAndExtra,
      ThinPrismFisheyeFocalAndExtra_num_alloc,
      delta,
      delta_num_alloc,
      out_ThinPrismFisheyeFocalAndExtra_retracted,
      out_ThinPrismFisheyeFocalAndExtra_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar