#include "kernel_thin_prism_fisheye_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeJtjnjtrDirectKernel(double* pose_njtr,
                                        unsigned int pose_njtr_num_alloc,
                                        SharedIndex* pose_njtr_indices,
                                        double* pose_jac,
                                        unsigned int pose_jac_num_alloc,
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
                                        double* const out_pose_njtr,
                                        unsigned int out_pose_njtr_num_alloc,
                                        double* const out_calib_njtr,
                                        unsigned int out_calib_njtr_num_alloc,
                                        double* const out_point_njtr,
                                        unsigned int out_point_njtr_num_alloc,
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

  __shared__ SharedIndex point_njtr_indices_loc[1024];
  point_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42;

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
    r4 = fma(r4, r6, r2);
  };
  LoadShared<2, double, double>(calib_njtr,
                                8 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 12 * calib_jac_num_alloc, global_thread_idx, r9, r10);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 10 * calib_jac_num_alloc, global_thread_idx, r11, r12);
  };
  LoadShared<2, double, double>(calib_njtr,
                                4 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r13,
                        r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 4 * calib_jac_num_alloc, global_thread_idx, r15, r16);
  };
  LoadShared<2, double, double>(calib_njtr,
                                6 * calib_njtr_num_alloc,
                                calib_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        calib_njtr_indices_loc[threadIdx.x].target,
                        r17,
                        r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 8 * calib_jac_num_alloc, global_thread_idx, r19, r20);
    ReadIdx2<1024, double, double, double2>(
        calib_jac, 6 * calib_jac_num_alloc, global_thread_idx, r21, r22);
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
    r4 = fma(r8, r9, r4);
    r4 = fma(r2, r11, r4);
    r4 = fma(r14, r15, r4);
    r4 = fma(r18, r19, r4);
    r4 = fma(r17, r21, r4);
    r4 = fma(r13, r23, r4);
    r4 = fma(r25, r27, r4);
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r25);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r29, r30);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r31,
                        r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r33, r34);
    r35 = fma(r32, r33, r25 * r29);
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r36, r37);
    r35 = fma(r31, r36, r35);
    r38 = r4 + r35;
    r5 = fma(r5, r7, r3);
    r5 = fma(r2, r12, r5);
    r5 = fma(r17, r22, r5);
    r5 = fma(r18, r20, r5);
    r5 = fma(r8, r10, r5);
    r5 = fma(r14, r16, r5);
    r5 = fma(r13, r24, r5);
    r5 = fma(r26, r28, r5);
    r32 = fma(r32, r34, r25 * r30);
    r32 = fma(r31, r37, r32);
    r31 = r5 + r32;
    r25 = fma(r1, r31, r0 * r38);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r26, r13);
    r14 = fma(r13, r31, r26 * r38);
    WriteSum2<double, double>((double*)inout_shared, r25, r14);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r14, r25);
    r8 = fma(r25, r31, r14 * r38);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r18, r17);
    r2 = fma(r17, r31, r18 * r38);
    WriteSum2<double, double>((double*)inout_shared, r8, r2);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r2, r8);
    r3 = fma(r8, r31, r2 * r38);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r39, r40);
    r31 = fma(r40, r31, r39 * r38);
    WriteSum2<double, double>((double*)inout_shared, r3, r31);
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
                        r31,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r31, r2, r3 * r39);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r39,
                        r38);
  };
  __syncthreads();
  LoadShared<2, double, double>(pose_njtr,
                                0 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r41,
                        r42);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r2 = fma(r39, r14, r2);
    r2 = fma(r38, r18, r2);
    r2 = fma(r41, r0, r2);
    r2 = fma(r42, r26, r2);
    r35 = r2 + r35;
    r27 = r27 * r35;
    r17 = fma(r38, r17, r3 * r40);
    r17 = fma(r39, r25, r17);
    r17 = fma(r41, r1, r17);
    r17 = fma(r42, r13, r17);
    r17 = fma(r31, r8, r17);
    r32 = r17 + r32;
    r28 = r28 * r32;
    WriteSum2<double, double>((double*)inout_shared, r27, r28);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            0 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<double, double>((double*)inout_shared, r35, r32);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            2 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fma(r24, r32, r23 * r35);
    r15 = fma(r15, r35, r16 * r32);
    WriteSum2<double, double>((double*)inout_shared, r24, r15);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            4 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r21 = fma(r21, r35, r22 * r32);
    r19 = fma(r19, r35, r20 * r32);
    WriteSum2<double, double>((double*)inout_shared, r21, r19);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            6 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fma(r11, r35, r12 * r32);
    r9 = fma(r9, r35, r10 * r32);
    WriteSum2<double, double>((double*)inout_shared, r11, r9);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            8 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = r6 * r35;
    r32 = r7 * r32;
    WriteSum2<double, double>((double*)inout_shared, r35, r32);
  };
  FlushSumShared<2, double>(out_calib_njtr,
                            10 * out_calib_njtr_num_alloc,
                            calib_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = r4 + r2;
    r17 = r5 + r17;
    r37 = fma(r37, r17, r36 * r2);
    r34 = fma(r34, r17, r33 * r2);
    WriteSum2<double, double>((double*)inout_shared, r37, r34);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = fma(r30, r17, r29 * r2);
    WriteSum1<double, double>((double*)inout_shared, r17);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeJtjnjtrDirect(double* pose_njtr,
                                   unsigned int pose_njtr_num_alloc,
                                   SharedIndex* pose_njtr_indices,
                                   double* pose_jac,
                                   unsigned int pose_jac_num_alloc,
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
                                   double* const out_pose_njtr,
                                   unsigned int out_pose_njtr_num_alloc,
                                   double* const out_calib_njtr,
                                   unsigned int out_calib_njtr_num_alloc,
                                   double* const out_point_njtr,
                                   unsigned int out_point_njtr_num_alloc,
                                   size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeJtjnjtrDirectKernel<<<n_blocks, 1024>>>(
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
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_calib_njtr,
      out_calib_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar