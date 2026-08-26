#include "kernel_ThinPrismFisheyeCalib_update_Mp.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ThinPrismFisheyeCalibUpdateMpKernel(
    double* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    double* ThinPrismFisheyeCalib_Mp,
    unsigned int ThinPrismFisheyeCalib_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeCalib_Mp_kp1,
    unsigned int out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_Mp,
        0 * ThinPrismFisheyeCalib_Mp_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        0 * ThinPrismFisheyeCalib_r_k_num_alloc,
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
        out_ThinPrismFisheyeCalib_Mp_kp1,
        0 * out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_Mp,
        2 * ThinPrismFisheyeCalib_Mp_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        2 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r5,
        r6);
    r3 = fma(r3, r4, r5);
    r2 = fma(r2, r4, r6);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_Mp_kp1,
        2 * out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_Mp,
        4 * ThinPrismFisheyeCalib_Mp_num_alloc,
        global_thread_idx,
        r6,
        r5);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        4 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r7,
        r8);
    r6 = fma(r6, r4, r7);
    r5 = fma(r5, r4, r8);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_Mp_kp1,
        4 * out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r6,
        r5);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_Mp,
        6 * ThinPrismFisheyeCalib_Mp_num_alloc,
        global_thread_idx,
        r8,
        r7);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        6 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r9,
        r10);
    r8 = fma(r8, r4, r9);
    r7 = fma(r7, r4, r10);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_Mp_kp1,
        6 * out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r8,
        r7);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_Mp,
        8 * ThinPrismFisheyeCalib_Mp_num_alloc,
        global_thread_idx,
        r10,
        r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        8 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r11,
        r12);
    r10 = fma(r10, r4, r11);
    r9 = fma(r9, r4, r12);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_Mp_kp1,
        8 * out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r10,
        r9);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_Mp,
        10 * ThinPrismFisheyeCalib_Mp_num_alloc,
        global_thread_idx,
        r12,
        r11);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_r_k,
        10 * ThinPrismFisheyeCalib_r_k_num_alloc,
        global_thread_idx,
        r13,
        r14);
    r12 = fma(r12, r4, r13);
    r4 = fma(r11, r4, r14);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_Mp_kp1,
        10 * out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
        global_thread_idx,
        r12,
        r4);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        0 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        2 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r3,
        r2);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        4 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r6,
        r5);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        6 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r8,
        r7);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        8 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r10,
        r9);
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        10 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r12,
        r4);
  };
}

void ThinPrismFisheyeCalibUpdateMp(
    double* ThinPrismFisheyeCalib_r_k,
    unsigned int ThinPrismFisheyeCalib_r_k_num_alloc,
    double* ThinPrismFisheyeCalib_Mp,
    unsigned int ThinPrismFisheyeCalib_Mp_num_alloc,
    const double* const beta,
    double* out_ThinPrismFisheyeCalib_Mp_kp1,
    unsigned int out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
    double* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibUpdateMpKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_r_k,
      ThinPrismFisheyeCalib_r_k_num_alloc,
      ThinPrismFisheyeCalib_Mp,
      ThinPrismFisheyeCalib_Mp_num_alloc,
      beta,
      out_ThinPrismFisheyeCalib_Mp_kp1,
      out_ThinPrismFisheyeCalib_Mp_kp1_num_alloc,
      out_ThinPrismFisheyeCalib_w,
      out_ThinPrismFisheyeCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar