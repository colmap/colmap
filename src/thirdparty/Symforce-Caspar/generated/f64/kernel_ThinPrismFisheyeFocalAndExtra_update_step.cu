#include "kernel_ThinPrismFisheyeFocalAndExtra_update_step.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraUpdateStepKernel(
        double* ThinPrismFisheyeFocalAndExtra_step_k,
        unsigned int ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_p_kp1,
        unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        const double* const alpha,
        double* out_ThinPrismFisheyeFocalAndExtra_step_kp1,
        unsigned int out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step_k,
        0 * ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        0 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  LoadUnique<1, double, double>(alpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_step_kp1,
        0 * out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step_k,
        2 * ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        2 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r1,
        r0);
    r1 = fma(r1, r4, r3);
    r0 = fma(r0, r4, r2);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_step_kp1,
        2 * out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step_k,
        4 * ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        4 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_step_kp1,
        4 * out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step_k,
        6 * ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        6 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r1,
        r0);
    r1 = fma(r1, r4, r3);
    r0 = fma(r0, r4, r2);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_step_kp1,
        6 * out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_step_k,
        8 * ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_p_kp1,
        8 * ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    r2 = fma(r2, r4, r0);
    r4 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_step_kp1,
        8 * out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
        global_thread_idx,
        r2,
        r4);
  };
}

void ThinPrismFisheyeFocalAndExtraUpdateStep(
    double* ThinPrismFisheyeFocalAndExtra_step_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_p_kp1,
    unsigned int ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
    const double* const alpha,
    double* out_ThinPrismFisheyeFocalAndExtra_step_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraUpdateStepKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeFocalAndExtra_step_k,
      ThinPrismFisheyeFocalAndExtra_step_k_num_alloc,
      ThinPrismFisheyeFocalAndExtra_p_kp1,
      ThinPrismFisheyeFocalAndExtra_p_kp1_num_alloc,
      alpha,
      out_ThinPrismFisheyeFocalAndExtra_step_kp1,
      out_ThinPrismFisheyeFocalAndExtra_step_kp1_num_alloc,
      problem_size);
}

}  // namespace caspar