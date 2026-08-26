#include "kernel_thin_prism_fisheye_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeFixedPointJtjnjtrDirectKernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* calib_njtr,
        unsigned int calib_njtr_num_alloc,
        SharedIndex* calib_njtr_indices,
        double* calib_jac,
        unsigned int calib_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_calib_njtr,
        unsigned int out_calib_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_njtr_indices_loc[1024];
  calib_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(calib_njtr,
                                2 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  LoadShared<2, double, double>(calib_njtr,
                                10 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r4,
                        r5);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 14 * calib_jac_num_alloc, global_thread_idx, r6, r7);
    r5 = fma(r5, r7, r3);
  };
  LoadShared<2, double, double>(calib_njtr,
                                8 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r3,
                        r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 10 * calib_jac_num_alloc, global_thread_idx, r9, r10);
  };
  LoadShared<2, double, double>(calib_njtr,
                                6 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r11,
                        r12);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 6 * calib_jac_num_alloc, global_thread_idx, r13, r14);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 8 * calib_jac_num_alloc, global_thread_idx, r15, r16);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 12 * calib_jac_num_alloc, global_thread_idx, r17, r18);
  };
  LoadShared<2, double, double>(calib_njtr,
                                4 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r19,
                        r20);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 4 * calib_jac_num_alloc, global_thread_idx, r21, r22);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 2 * calib_jac_num_alloc, global_thread_idx, r23, r24);
  };
  LoadShared<2, double, double>(calib_njtr,
                                0 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r25,
                        r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 0 * calib_jac_num_alloc, global_thread_idx, r27, r28);
    r5 = fma(r3, r10, r5);
    r5 = fma(r11, r14, r5);
    r5 = fma(r12, r16, r5);
    r5 = fma(r8, r18, r5);
    r5 = fma(r20, r22, r5);
    r5 = fma(r19, r24, r5);
    r5 = fma(r26, r28, r5);
    r4 = fma(r4, r6, r2);
    r4 = fma(r8, r17, r4);
    r4 = fma(r3, r9, r4);
    r4 = fma(r20, r21, r4);
    r4 = fma(r12, r15, r4);
    r4 = fma(r11, r13, r4);
    r4 = fma(r19, r23, r4);
    r4 = fma(r25, r27, r4);
    r25 = fma(r0, r4, r1 * r5);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r19, r11);
    r12 = fma(r19, r4, r11 * r5);
    WriteSum2<double, double>((double*)inout_shared, r25, r12);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r12, r25);
    r20 = fma(r12, r4, r25 * r5);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r3, r8);
    r2 = fma(r3, r4, r8 * r5);
    WriteSum2<double, double>((double*)inout_shared, r20, r2);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r2, r20);
    r26 = fma(r2, r4, r20 * r5);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r29, r30);
    r4 = fma(r29, r4, r30 * r5);
    WriteSum2<double, double>((double*)inout_shared, r26, r4);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  LoadShared<2, double, double>(pose_njtr,
                                4 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r4,
                        r26);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r4, r2, r26 * r29);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r29,
                        r5);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr,
                                0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r31,
                        r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r29, r12, r2);
    r2 = fma(r5, r3, r2);
    r2 = fma(r31, r0, r2);
    r2 = fma(r32, r19, r2);
    r27 = r27 * r2;
    r8 = fma(r5, r8, r26 * r30);
    r8 = fma(r29, r25, r8);
    r8 = fma(r31, r1, r8);
    r8 = fma(r32, r11, r8);
    r8 = fma(r4, r20, r8);
    r28 = r28 * r8;
    WriteSum2<double, double>((double*)inout_shared, r27, r28);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r2, r8);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r24, r8, r23 * r2);
    r21 = fma(r21, r2, r22 * r8);
    WriteSum2<double, double>((double*)inout_shared, r24, r21);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fma(r13, r2, r14 * r8);
    r15 = fma(r15, r2, r16 * r8);
    WriteSum2<double, double>((double*)inout_shared, r13, r15);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fma(r9, r2, r10 * r8);
    r17 = fma(r17, r2, r18 * r8);
    WriteSum2<double, double>((double*)inout_shared, r9, r17);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r6 * r2;
    r8 = r7 * r8;
    WriteSum2<double, double>((double*)inout_shared, r2, r8);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeFixedPointJtjnjtrDirect(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* calib_njtr,
    unsigned int calib_njtr_num_alloc,
    SharedIndex* calib_njtr_indices,
    double* calib_jac,
    unsigned int calib_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeFixedPointJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      calib_njtr,
      calib_njtr_num_alloc,
      calib_njtr_indices,
      calib_jac,
      calib_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar