#include "kernel_ThinPrismFisheyeCalib_start_w.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ThinPrismFisheyeCalibStartWKernel(
    double* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyeCalib_p,
    unsigned int ThinPrismFisheyeCalib_p_num_alloc,
    double* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[8192];

  double r0, r1, r2, r3, r4;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        0 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
  };
  LoadUnique<1, double, double>(diag, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>((double*)inout_shared, 0, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r0 = r0 * r2;
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p,
        0 * ThinPrismFisheyeCalib_p_num_alloc,
        global_thread_idx,
        r3,
        r4);
    r0 = r0 * r3;
    r1 = r1 * r2;
    r1 = r1 * r4;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        0 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        2 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r1,
        r0);
    r1 = r1 * r2;
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p,
        2 * ThinPrismFisheyeCalib_p_num_alloc,
        global_thread_idx,
        r4,
        r3);
    r1 = r1 * r4;
    r0 = r0 * r2;
    r0 = r0 * r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        2 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        4 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
    r0 = r0 * r2;
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p,
        4 * ThinPrismFisheyeCalib_p_num_alloc,
        global_thread_idx,
        r3,
        r4);
    r0 = r0 * r3;
    r1 = r1 * r2;
    r1 = r1 * r4;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        4 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        6 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r1,
        r0);
    r1 = r1 * r2;
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p,
        6 * ThinPrismFisheyeCalib_p_num_alloc,
        global_thread_idx,
        r4,
        r3);
    r1 = r1 * r4;
    r0 = r0 * r2;
    r0 = r0 * r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        6 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        8 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r0,
        r1);
    r0 = r0 * r2;
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p,
        8 * ThinPrismFisheyeCalib_p_num_alloc,
        global_thread_idx,
        r3,
        r4);
    r0 = r0 * r3;
    r1 = r1 * r2;
    r1 = r1 * r4;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        8 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r0,
        r1);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_precond_diag,
        10 * ThinPrismFisheyeCalib_precond_diag_num_alloc,
        global_thread_idx,
        r1,
        r0);
    r1 = r1 * r2;
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib_p,
        10 * ThinPrismFisheyeCalib_p_num_alloc,
        global_thread_idx,
        r4,
        r3);
    r1 = r1 * r4;
    r2 = r0 * r2;
    r2 = r2 * r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_w,
        10 * out_ThinPrismFisheyeCalib_w_num_alloc,
        global_thread_idx,
        r1,
        r2);
  };
}

void ThinPrismFisheyeCalibStartW(
    double* ThinPrismFisheyeCalib_precond_diag,
    unsigned int ThinPrismFisheyeCalib_precond_diag_num_alloc,
    const double* const diag,
    double* ThinPrismFisheyeCalib_p,
    unsigned int ThinPrismFisheyeCalib_p_num_alloc,
    double* out_ThinPrismFisheyeCalib_w,
    unsigned int out_ThinPrismFisheyeCalib_w_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibStartWKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib_precond_diag,
      ThinPrismFisheyeCalib_precond_diag_num_alloc,
      diag,
      ThinPrismFisheyeCalib_p,
      ThinPrismFisheyeCalib_p_num_alloc,
      out_ThinPrismFisheyeCalib_w,
      out_ThinPrismFisheyeCalib_w_num_alloc,
      problem_size);
}

}  // namespace caspar