#include "kernel_thin_prism_fisheye_split_fixed_principal_point_jtjnjtr_direct.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    ThinPrismFisheyeSplitFixedPrincipalPointJtjnjtrDirectKernel(
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
        double* point_njtr,
        unsigned int point_njtr_num_alloc,
        SharedIndex* point_njtr_indices,
        double* point_jac,
        unsigned int point_jac_num_alloc,
        double* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        double* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_njtr_indices_loc[1024];
  focal_and_extra_njtr_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_njtr_indices[global_thread_idx]
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
  };
  LoadShared<1, double, double>(point_njtr,
                                2 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, point_njtr_indices_loc[threadIdx.x].target, r24);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 4 * point_jac_num_alloc, global_thread_idx, r14, r29);
  };
  LoadShared<2, double, double>(point_njtr,
                                0 * point_njtr_num_alloc,
                                point_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        point_njtr_indices_loc[threadIdx.x].target,
                        r30,
                        r31);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        point_jac, 2 * point_jac_num_alloc, global_thread_idx, r32, r33);
    r34 = fma(r31, r33, r24 * r29);
    ReadIdx2<1024, double, double, double2>(
        point_jac, 0 * point_jac_num_alloc, global_thread_idx, r35, r36);
    r34 = fma(r30, r36, r34);
    r37 = r8 + r34;
    r9 = fma(r9, r11, r3 * r4);
    r9 = fma(r13, r15, r9);
    r9 = fma(r2, r6, r9);
    r9 = fma(r23, r25, r9);
    r9 = fma(r10, r27, r9);
    r9 = fma(r18, r19, r9);
    r9 = fma(r17, r21, r9);
    r31 = fma(r31, r32, r24 * r14);
    r31 = fma(r30, r35, r31);
    r30 = r9 + r31;
    r24 = fma(r0, r30, r1 * r37);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 2 * pose_jac_num_alloc, global_thread_idx, r17, r18);
    r10 = fma(r17, r30, r18 * r37);
    WriteSum2<double, double>((double*)inout_shared, r24, r10);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 4 * pose_jac_num_alloc, global_thread_idx, r10, r24);
    r23 = fma(r10, r30, r24 * r37);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 6 * pose_jac_num_alloc, global_thread_idx, r2, r13);
    r3 = fma(r2, r30, r13 * r37);
    WriteSum2<double, double>((double*)inout_shared, r23, r3);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 8 * pose_jac_num_alloc, global_thread_idx, r3, r23);
    r38 = fma(r3, r30, r23 * r37);
    ReadIdx2<1024, double, double, double2>(
        pose_jac, 10 * pose_jac_num_alloc, global_thread_idx, r39, r40);
    r30 = fma(r39, r30, r40 * r37);
    WriteSum2<double, double>((double*)inout_shared, r38, r30);
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
                        r30,
                        r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r3 = fma(r30, r3, r38 * r39);
  };
  LoadShared<2, double, double>(pose_njtr,
                                2 * pose_njtr_num_alloc,
                                pose_njtr_indices_loc,
                                (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared,
                        pose_njtr_indices_loc[threadIdx.x].target,
                        r39,
                        r37);
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
    r3 = fma(r39, r10, r3);
    r3 = fma(r37, r2, r3);
    r3 = fma(r41, r0, r3);
    r3 = fma(r42, r17, r3);
    r31 = r3 + r31;
    r15 = r15 * r31;
    r13 = fma(r37, r13, r38 * r40);
    r13 = fma(r39, r24, r13);
    r13 = fma(r41, r1, r13);
    r13 = fma(r42, r18, r13);
    r13 = fma(r30, r23, r13);
    r34 = r13 + r34;
    r16 = r16 * r34;
    WriteSum2<double, double>((double*)inout_shared, r15, r16);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            0 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = fma(r7, r34, r6 * r31);
    r5 = fma(r5, r34, r4 * r31);
    WriteSum2<double, double>((double*)inout_shared, r7, r5);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            2 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r12 = fma(r12, r34, r11 * r31);
    r28 = fma(r28, r34, r27 * r31);
    WriteSum2<double, double>((double*)inout_shared, r12, r28);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            4 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fma(r22, r34, r21 * r31);
    r20 = fma(r20, r34, r19 * r31);
    WriteSum2<double, double>((double*)inout_shared, r22, r20);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            6 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = r25 * r31;
    r34 = r26 * r34;
    WriteSum2<double, double>((double*)inout_shared, r31, r34);
  };
  FlushSumShared<2, double>(out_focal_and_extra_njtr,
                            8 * out_focal_and_extra_njtr_num_alloc,
                            focal_and_extra_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = r8 + r13;
    r3 = r9 + r3;
    r35 = fma(r35, r3, r36 * r13);
    r32 = fma(r32, r3, r33 * r13);
    WriteSum2<double, double>((double*)inout_shared, r35, r32);
  };
  FlushSumShared<2, double>(out_point_njtr,
                            0 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r3 = fma(r14, r3, r29 * r13);
    WriteSum1<double, double>((double*)inout_shared, r3);
  };
  FlushSumShared<1, double>(out_point_njtr,
                            2 * out_point_njtr_num_alloc,
                            point_njtr_indices_loc,
                            (double*)inout_shared);
}

void ThinPrismFisheyeSplitFixedPrincipalPointJtjnjtrDirect(
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
    double* point_njtr,
    unsigned int point_njtr_num_alloc,
    SharedIndex* point_njtr_indices,
    double* point_jac,
    unsigned int point_jac_num_alloc,
    double* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    double* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    double* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  ThinPrismFisheyeSplitFixedPrincipalPointJtjnjtrDirectKernel<<<n_blocks,
                                                                1024>>>(
      pose_njtr,
      pose_njtr_num_alloc,
      pose_njtr_indices,
      pose_jac,
      pose_jac_num_alloc,
      focal_and_extra_njtr,
      focal_and_extra_njtr_num_alloc,
      focal_and_extra_njtr_indices,
      focal_and_extra_jac,
      focal_and_extra_jac_num_alloc,
      point_njtr,
      point_njtr_num_alloc,
      point_njtr_indices,
      point_jac,
      point_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      problem_size);
}

}  // namespace caspar