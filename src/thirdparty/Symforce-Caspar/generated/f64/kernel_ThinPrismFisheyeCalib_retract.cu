#include "kernel_ThinPrismFisheyeCalib_retract.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) ThinPrismFisheyeCalibRetractKernel(
    double* ThinPrismFisheyeCalib,
    unsigned int ThinPrismFisheyeCalib_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_ThinPrismFisheyeCalib_retracted,
    unsigned int out_ThinPrismFisheyeCalib_retracted_num_alloc,
    size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  double r0, r1, r2, r3;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(ThinPrismFisheyeCalib,
                                            0 * ThinPrismFisheyeCalib_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(
        delta, 0 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_retracted,
        0 * out_ThinPrismFisheyeCalib_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(ThinPrismFisheyeCalib,
                                            2 * ThinPrismFisheyeCalib_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r2);
    ReadIdx2<1024, double, double, double2>(
        delta, 2 * delta_num_alloc, global_thread_idx, r1, r0);
    r1 = r3 + r1;
    r0 = r2 + r0;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_retracted,
        2 * out_ThinPrismFisheyeCalib_retracted_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(ThinPrismFisheyeCalib,
                                            4 * ThinPrismFisheyeCalib_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(
        delta, 4 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_retracted,
        4 * out_ThinPrismFisheyeCalib_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(ThinPrismFisheyeCalib,
                                            6 * ThinPrismFisheyeCalib_num_alloc,
                                            global_thread_idx,
                                            r3,
                                            r2);
    ReadIdx2<1024, double, double, double2>(
        delta, 6 * delta_num_alloc, global_thread_idx, r1, r0);
    r1 = r3 + r1;
    r0 = r2 + r0;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_retracted,
        6 * out_ThinPrismFisheyeCalib_retracted_num_alloc,
        global_thread_idx,
        r1,
        r0);
    ReadIdx2<1024, double, double, double2>(ThinPrismFisheyeCalib,
                                            8 * ThinPrismFisheyeCalib_num_alloc,
                                            global_thread_idx,
                                            r0,
                                            r1);
    ReadIdx2<1024, double, double, double2>(
        delta, 8 * delta_num_alloc, global_thread_idx, r2, r3);
    r2 = r0 + r2;
    r3 = r1 + r3;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_retracted,
        8 * out_ThinPrismFisheyeCalib_retracted_num_alloc,
        global_thread_idx,
        r2,
        r3);
    ReadIdx2<1024, double, double, double2>(
        ThinPrismFisheyeCalib,
        10 * ThinPrismFisheyeCalib_num_alloc,
        global_thread_idx,
        r3,
        r2);
    ReadIdx2<1024, double, double, double2>(
        delta, 10 * delta_num_alloc, global_thread_idx, r1, r0);
    r1 = r3 + r1;
    r0 = r2 + r0;
    WriteIdx2<1024, double, double, double2>(
        out_ThinPrismFisheyeCalib_retracted,
        10 * out_ThinPrismFisheyeCalib_retracted_num_alloc,
        global_thread_idx,
        r1,
        r0);
  };
}

void ThinPrismFisheyeCalibRetract(
    double* ThinPrismFisheyeCalib,
    unsigned int ThinPrismFisheyeCalib_num_alloc,
    double* delta,
    unsigned int delta_num_alloc,
    double* out_ThinPrismFisheyeCalib_retracted,
    unsigned int out_ThinPrismFisheyeCalib_retracted_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeCalibRetractKernel<<<n_blocks, 1024>>>(
      ThinPrismFisheyeCalib,
      ThinPrismFisheyeCalib_num_alloc,
      delta,
      delta_num_alloc,
      out_ThinPrismFisheyeCalib_retracted,
      out_ThinPrismFisheyeCalib_retracted_num_alloc,
      problem_size);
}

}  // namespace caspar