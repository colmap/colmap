#include "kernel_thin_prism_fisheye_split_fixed_principal_point_fixed_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel(
        double* pose_njtr,
        unsigned int pose_njtr_num_alloc,
        SharedIndex* pose_njtr_indices,
        double* pose_jac,
        unsigned int pose_jac_num_alloc,
        double* focal_and_extra_njtr,
        unsigned int focal_and_extra_njtr_num_alloc,
        SharedIndex* focal_and_extra_njtr_indices,
        double* focal_and_extra_jac,
        unsigned int focal_and_extra_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_njtr_indices_loc[1024];
  pose_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 0 * pose_jac_num_alloc, global_thread_idx, r0, r1);
  };
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                2 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r2,
                        r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            4 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r4,
                                            r5);
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            2 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r6,
                                            r7);
    r8 = fma(r2, r7, r3 * r5);
  };
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                4 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r9,
                        r10);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            6 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r11,
                                            r12);
  };
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                0 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r13,
                        r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            0 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r15,
                                            r16);
  };
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                6 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r17,
                        r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            12 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r19,
                                            r20);
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            10 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r21,
                                            r22);
  };
  LoadShared<2, double, double>(focal_and_extra_njtr,
                                8 * focal_and_extra_njtr_num_alloc,
                                focal_and_extra_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        focal_and_extra_njtr_indices_loc[threadIdx.x].target,
                        r23,
                        r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            14 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r25,
                                            r26);
    ReadIdx2<1024, double, double, double2>(focal_and_extra_jac,
                                            8 * focal_and_extra_jac_num_alloc,
                                            global_thread_idx,
                                            r27,
                                            r28);
    r8 = fma(r9, r12, r8);
    r8 = fma(r14, r16, r8);
    r8 = fma(r18, r20, r8);
    r8 = fma(r17, r22, r8);
    r8 = fma(r24, r26, r8);
    r8 = fma(r10, r28, r8);
    r9 = fma(r9, r11, r3 * r4);
    r9 = fma(r13, r15, r9);
    r9 = fma(r2, r6, r9);
    r9 = fma(r23, r25, r9);
    r9 = fma(r10, r27, r9);
    r9 = fma(r18, r19, r9);
    r9 = fma(r17, r21, r9);
    r17 = fma(r0, r9, r1 * r8);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r18, r10);
    r23 = fma(r18, r9, r10 * r8);
    WriteSum2<double, double>((double*)inout_shared, r17, r23);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r23, r17);
    r2 = fma(r23, r9, r17 * r8);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r13, r3);
    r24 = fma(r13, r9, r3 * r8);
    WriteSum2<double, double>((double*)inout_shared, r2, r24);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r24, r2);
    r14 = fma(r24, r9, r2 * r8);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r29, r30);
    r9 = fma(r29, r9, r30 * r8);
    WriteSum2<double, double>((double*)inout_shared, r14, r9);
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
                        r9,
                        r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r24 = fma(r9, r24, r14 * r29);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r29,
                        r8);
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
    r24 = fma(r29, r23, r24);
    r24 = fma(r8, r13, r24);
    r24 = fma(r31, r0, r24);
    r24 = fma(r32, r18, r24);
    r15 = r15 * r24;
    r3 = fma(r8, r3, r14 * r30);
    r3 = fma(r29, r17, r3);
    r3 = fma(r31, r1, r3);
    r3 = fma(r32, r10, r3);
    r3 = fma(r9, r2, r3);
    r16 = r16 * r3;
    WriteSum2<double, double>((double*)inout_shared, r15, r16);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r7, r3, r6 * r24);
    r5 = fma(r5, r3, r4 * r24);
    WriteSum2<double, double>((double*)inout_shared, r7, r5);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r12, r3, r11 * r24);
    r28 = fma(r28, r3, r27 * r24);
    WriteSum2<double, double>((double*)inout_shared, r12, r28);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r22, r3, r21 * r24);
    r20 = fma(r20, r3, r19 * r24);
    WriteSum2<double, double>((double*)inout_shared, r22, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = r25 * r24;
    r3 = r26 * r3;
    WriteSum2<double, double>((double*)inout_shared, r24, r3);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointFixedPointJtjnjtrDirect(
    double* pose_njtr,
    unsigned int pose_njtr_num_alloc,
    SharedIndex* pose_njtr_indices,
    double* pose_jac,
    unsigned int pose_jac_num_alloc,
    double* focal_and_extra_njtr,
    unsigned int focal_and_extra_njtr_num_alloc,
    SharedIndex* focal_and_extra_njtr_indices,
    double* focal_and_extra_jac,
    unsigned int focal_and_extra_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointFixedPointJtjnjtrDirectKernel<<<
      n_blocks,
      1024>>>(pose_njtr,
              pose_njtr_num_alloc,
              pose_njtr_indices,
              pose_jac,
              pose_jac_num_alloc,
              focal_and_extra_njtr,
              focal_and_extra_njtr_num_alloc,
              focal_and_extra_njtr_indices,
              focal_and_extra_jac,
              focal_and_extra_jac_num_alloc,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_focal_and_extra_njtr,
              out_focal_and_extra_njtr_num_alloc,
              problem_size);
}

}  // namespace caspar