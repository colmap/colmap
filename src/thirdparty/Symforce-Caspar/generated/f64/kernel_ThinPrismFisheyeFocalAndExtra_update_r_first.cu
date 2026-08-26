#include "kernel_ThinPrismFisheyeFocalAndExtra_update_r_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraUpdateRFirstKernel(
        double* ThinPrismFisheyeFocalAndExtra_r_k,
        unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_w,
        unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        const double* const negalpha,
        double* out_ThinPrismFisheyeFocalAndExtra_r_kp1,
        unsigned int out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
        double* const out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot,
        double* const out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  __shared__ double out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot_local[1];

  __shared__ double out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        0 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        0 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  LoadUnique<1, double, double>(negalpha, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r2, r4, r0);
    r3 = fma(r3, r4, r1);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_r_kp1,
        0 * out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        2 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r5,
        r6);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        2 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r7,
        r8);
    r7 = fma(r7, r4, r5);
    r8 = fma(r8, r4, r6);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_r_kp1,
        2 * out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
        global_thread_idx,
        r7,
        r8);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        4 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r9,
        r10);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        4 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r11 = fma(r11, r4, r9);
    r12 = fma(r12, r4, r10);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_r_kp1,
        4 * out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
        global_thread_idx,
        r11,
        r12);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        6 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r13,
        r14);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        6 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r15,
        r16);
    r15 = fma(r15, r4, r13);
    r16 = fma(r16, r4, r14);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_r_kp1,
        6 * out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
        global_thread_idx,
        r15,
        r16);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        8 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r17,
        r18);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_w,
        8 * ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r19,
        r20);
    r19 = fma(r19, r4, r17);
    r4 = fma(r20, r4, r18);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_r_kp1,
        8 * out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
        global_thread_idx,
        r19,
        r4);
    r6 = fma(r6, r6, r0 * r0);
    r6 = fma(r10, r10, r6);
    r6 = fma(r17, r17, r6);
    r6 = fma(r1, r1, r6);
    r6 = fma(r9, r9, r6);
    r6 = fma(r14, r14, r6);
    r6 = fma(r5, r5, r6);
    r6 = fma(r13, r13, r6);
    r6 = fma(r18, r18, r6);
  };
  SumStore<double>(out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r6);
  if (global_thread_idx < problem_size) {
    r16 = fma(r16, r16, r19 * r19);
    r16 = fma(r2, r2, r16);
    r16 = fma(r7, r7, r16);
    r16 = fma(r12, r12, r16);
    r16 = fma(r8, r8, r16);
    r16 = fma(r15, r15, r16);
    r16 = fma(r3, r3, r16);
    r16 = fma(r11, r11, r16);
    r16 = fma(r4, r4, r16);
  };
  SumStore<double>(out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r16);
  SumFlushFinal<double>(out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot_local,
                        out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot,
                        1);
  SumFlushFinal<double>(out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot_local,
                        out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot,
                        1);
}

void ThinPrismFisheyeFocalAndExtraUpdateRFirst(
    double* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_w,
    unsigned int ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    const double* const negalpha,
    double* out_ThinPrismFisheyeFocalAndExtra_r_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
    double* const out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot,
    double* const out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraUpdateRFirstKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeFocalAndExtra_r_k,
      ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
      ThinPrismFisheyeFocalAndExtra_w,
      ThinPrismFisheyeFocalAndExtra_w_num_alloc,
      negalpha,
      out_ThinPrismFisheyeFocalAndExtra_r_kp1,
      out_ThinPrismFisheyeFocalAndExtra_r_kp1_num_alloc,
      out_ThinPrismFisheyeFocalAndExtra_r_0_norm2_tot,
      out_ThinPrismFisheyeFocalAndExtra_r_kp1_norm2_tot,
      problem_size);
}

}  // namespace caspar