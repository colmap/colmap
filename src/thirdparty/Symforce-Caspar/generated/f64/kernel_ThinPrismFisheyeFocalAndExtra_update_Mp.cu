#include "kernel_ThinPrismFisheyeFocalAndExtra_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFocalAndExtraUpdateMpKernel(
        double* ThinPrismFisheyeFocalAndExtra_r_k,
        unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        double* ThinPrismFisheyeFocalAndExtra_Mp,
        unsigned int ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
        const double* const beta,
        double* out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
        unsigned int out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
        double* out_ThinPrismFisheyeFocalAndExtra_w,
        unsigned int out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_Mp,
        0 * ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        0 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r2,
        r3);
  };
  LoadUnique<1, double, double>(beta, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = fma(r0, r4, r2);
    r1 = fma(r1, r4, r3);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
        0 * out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_Mp,
        2 * ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        2 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r5,
        r6);
    r3 = fma(r3, r4, r5);
    r2 = fma(r2, r4, r6);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
        2 * out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_Mp,
        4 * ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
        global_thread_idx,
        r6,
        r5);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        4 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r7,
        r8);
    r6 = fma(r6, r4, r7);
    r5 = fma(r5, r4, r8);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
        4 * out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
        global_thread_idx,
        r6,
        r5);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_Mp,
        6 * ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
        global_thread_idx,
        r8,
        r7);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        6 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r9,
        r10);
    r8 = fma(r8, r4, r9);
    r7 = fma(r7, r4, r10);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
        6 * out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
        global_thread_idx,
        r8,
        r7);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_Mp,
        8 * ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
        global_thread_idx,
        r10,
        r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeFocalAndExtra_r_k,
        8 * ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r10 = fma(r10, r4, r11);
    r4 = fma(r9, r4, r12);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
        8 * out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
        global_thread_idx,
        r10,
        r4);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_w,
        0 * out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_w,
        2 * out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r3,
        r2);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_w,
        4 * out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r6,
        r5);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_w,
        6 * out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r8,
        r7);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeFocalAndExtra_w,
        8 * out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
        global_thread_idx,
        r10,
        r4);
  };
}

void ThinPrismFisheyeFocalAndExtraUpdateMp(
    double* ThinPrismFisheyeFocalAndExtra_r_k,
    unsigned int ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
    double* ThinPrismFisheyeFocalAndExtra_Mp,
    unsigned int ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyeFocalAndExtra_w,
    unsigned int out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFocalAndExtraUpdateMpKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeFocalAndExtra_r_k,
      ThinPrismFisheyeFocalAndExtra_r_k_num_alloc,
      ThinPrismFisheyeFocalAndExtra_Mp,
      ThinPrismFisheyeFocalAndExtra_Mp_num_alloc,
      beta,
      out_ThinPrismFisheyeFocalAndExtra_Mp_kp1,
      out_ThinPrismFisheyeFocalAndExtra_Mp_kp1_num_alloc,
      out_ThinPrismFisheyeFocalAndExtra_w,
      out_ThinPrismFisheyeFocalAndExtra_w_num_alloc,
      problem_size);
}

}  // namespace caspar