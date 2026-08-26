#include "kernel_thin_prism_fisheye_fixed_pose_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPoseJtjnjtrDirectKernel(
        double* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        double* calib_jac,
        unsigned int calib_jac_num_alloc,
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        double* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r2);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r3, r4);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r5,
                        r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r7, r8);
    r9 = fma(r6, r7, r2 * r3);
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r10, r11);
    r9 = fma(r5, r10, r9);
    r12 = r0 * r9;
    r6 = fma(r6, r8, r2 * r4);
    r6 = fma(r5, r11, r6);
    r5 = r1 * r6;
    WriteSum2<double, double>((double*)inout_shared, r12, r5);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r9, r6);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 2 * calib_jac_num_alloc, global_thread_idx, r5, r12);
    r2 = fma(r5, r9, r12 * r6);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 4 * calib_jac_num_alloc, global_thread_idx, r13, r14);
    r15 = fma(r14, r6, r13 * r9);
    WriteSum2<double, double>((double*)inout_shared, r2, r15);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 6 * calib_jac_num_alloc, global_thread_idx, r15, r2);
    r16 = fma(r2, r6, r15 * r9);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 8 * calib_jac_num_alloc, global_thread_idx, r17, r18);
    r19 = fma(r18, r6, r17 * r9);
    WriteSum2<double, double>((double*)inout_shared, r16, r19);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 10 * calib_jac_num_alloc, global_thread_idx, r19, r16);
    r20 = fma(r16, r6, r19 * r9);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 12 * calib_jac_num_alloc, global_thread_idx, r21, r22);
    r23 = fma(r22, r6, r21 * r9);
    WriteSum2<double, double>((double*)inout_shared, r20, r23);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 14 * calib_jac_num_alloc, global_thread_idx, r23, r20);
    r9 = r23 * r9;
    r6 = r20 * r6;
    WriteSum2<double, double>((double*)inout_shared, r9, r6);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(calib_njtr,
                                2 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r6,
                        r9);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                10 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r24,
                        r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r24, r23, r6);
  };
  LoadShared<2, double, double>(calib_njtr,
                                8 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r24,
                        r6);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                4 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r26,
                        r27);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                6 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r28,
                        r29);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                0 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r30,
                        r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = fma(r6, r21, r23);
    r23 = fma(r24, r19, r23);
    r23 = fma(r27, r13, r23);
    r23 = fma(r29, r17, r23);
    r23 = fma(r28, r15, r23);
    r23 = fma(r26, r5, r23);
    r23 = fma(r30, r0, r23);
    r20 = fma(r25, r20, r9);
    r20 = fma(r24, r16, r20);
    r20 = fma(r28, r2, r20);
    r20 = fma(r29, r18, r20);
    r20 = fma(r6, r22, r20);
    r20 = fma(r27, r14, r20);
    r20 = fma(r26, r12, r20);
    r20 = fma(r31, r1, r20);
    r11 = fma(r11, r20, r10 * r23);
    r8 = fma(r8, r20, r7 * r23);
    WriteSum2<double, double>((double*)inout_shared, r11, r8);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r20 = fma(r4, r20, r3 * r23);
    WriteSum1<double, double>((double*)inout_shared, r20);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeFixedPoseJtjnjtrDirect(
    double* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    double* calib_jac,
    unsigned int calib_jac_num_alloc,
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPoseJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar