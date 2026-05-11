#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacFirstKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *pixel, unsigned int pixel_num_alloc, float *focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *point,
        unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_rTr,
        float *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx, r0, r5);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r6, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r9, r10, r11);
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r12, r13, r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r12 * r13;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = -2.00000000000000000e+00;
    r19 = r14 * r18;
    r20 = r15 * r19;
    r21 = r16 + r20;
    r6 = fmaf(r10, r21, r6);
    r22 = r12 * r14;
    r22 = r22 * r17;
    r23 = r13 * r15;
    r23 = r23 * r17;
    r24 = r22 + r23;
    r25 = r14 * r19;
    r26 = 1.00000000000000000e+00;
    r27 = r13 * r13;
    r28 = fmaf(r18, r27, r26);
    r29 = r25 + r28;
    r6 = fmaf(r11, r24, r6);
    r6 = fmaf(r9, r29, r6);
    r29 = r0 * r6;
    r30 = r14 * r15;
    r30 = r30 * r17;
    r16 = r16 + r30;
    r7 = fmaf(r9, r16, r7);
    r31 = r13 * r14;
    r31 = r31 * r17;
    r32 = r12 * r15;
    r32 = r32 * r18;
    r33 = r31 + r32;
    r25 = r26 + r25;
    r34 = r12 * r12;
    r35 = r18 * r34;
    r25 = r25 + r35;
    r7 = fmaf(r11, r33, r7);
    r7 = fmaf(r10, r25, r7);
    r25 = r7 * r7;
    r36 = 9.99999999999999955e-07;
    r37 = r12 * r15;
    r37 = r37 * r17;
    r31 = r31 + r37;
    r8 = fmaf(r10, r31, r8);
    r38 = r13 * r15;
    r38 = r38 * r18;
    r22 = r22 + r38;
    r28 = r35 + r28;
    r8 = fmaf(r9, r22, r8);
    r8 = fmaf(r11, r28, r8);
    r28 = copysign(1.0, r8);
    r28 = fmaf(r36, r28, r8);
    r36 = r28 * r28;
    r8 = 1.0 / r36;
    r35 = r6 * r6;
    r35 = fmaf(r8, r35, r8 * r25);
    r35 = fmaf(r5, r35, r26);
    r26 = 1.0 / r28;
    r25 = r35 * r26;
    r2 = fmaf(r29, r25, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r26;
    r1 = r1 * r35;
    r3 = fmaf(r7, r1, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r25 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r25);
  if (global_thread_idx < problem_size) {
    r25 = r4 * r3;
    r39 = r14 * r14;
    r15 = r15 * r15;
    r40 = r4 * r15;
    r41 = r39 + r40;
    r42 = r4 * r27;
    r43 = r34 + r42;
    r44 = r41 + r43;
    r44 = fmaf(r11, r44, r10 * r33);
    r33 = r17 * r7;
    r33 = r33 * r44;
    r45 = r18 * r7;
    r46 = r13 * r19;
    r32 = r32 + r46;
    r42 = r39 + r42;
    r39 = r4 * r34;
    r47 = r15 + r39;
    r42 = r42 + r47;
    r42 = fmaf(r10, r42, r11 * r32);
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r45 = r45 * r7;
    r45 = r45 * r42;
    r45 = fmaf(r28, r45, r8 * r33);
    r33 = r18 * r6;
    r33 = r33 * r6;
    r33 = r33 * r42;
    r45 = fmaf(r28, r33, r45);
    r32 = r17 * r6;
    r13 = r12 * r13;
    r13 = r13 * r18;
    r30 = r30 + r13;
    r24 = fmaf(r10, r24, r11 * r30);
    r32 = r32 * r24;
    r45 = fmaf(r8, r32, r45);
    r32 = r0 * r45;
    r33 = r5 * r7;
    r32 = r32 * r26;
    r32 = fmaf(r33, r32, r44 * r1);
    r44 = r0 * r4;
    r44 = r44 * r7;
    r44 = r44 * r8;
    r44 = r44 * r35;
    r32 = fmaf(r42, r44, r32);
    r30 = r4 * r2;
    r48 = r5 * r45;
    r48 = r48 * r26;
    r48 = fmaf(r29, r48, r24 * r1);
    r24 = r42 * r29;
    r35 = r4 * r35;
    r35 = r35 * r8;
    r48 = fmaf(r35, r24, r48);
    r30 = fmaf(r48, r30, r32 * r25);
    r25 = r4 * r2;
    r19 = r12 * r19;
    r38 = r38 + r19;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r15 = r15 + r14;
    r15 = r15 + r43;
    r15 = fmaf(r11, r15, r9 * r38);
    r40 = r34 + r40;
    r14 = r27 + r14;
    r40 = r40 + r14;
    r40 = fmaf(r9, r40, r11 * r22);
    r22 = r40 * r29;
    r22 = fmaf(r35, r22, r15 * r1);
    r34 = r17 * r7;
    r46 = r37 + r46;
    r46 = fmaf(r9, r46, r11 * r16);
    r34 = r34 * r46;
    r16 = r17 * r6;
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r34);
    r34 = r18 * r6;
    r34 = r34 * r6;
    r34 = r34 * r40;
    r16 = fmaf(r28, r34, r16);
    r15 = r18 * r7;
    r15 = r15 * r7;
    r15 = r15 * r40;
    r16 = fmaf(r28, r15, r16);
    r15 = r5 * r16;
    r15 = r15 * r26;
    r22 = fmaf(r29, r15, r22);
    r15 = r4 * r3;
    r46 = fmaf(r46, r1, r40 * r44);
    r34 = r0 * r16;
    r34 = r34 * r26;
    r46 = fmaf(r33, r34, r46);
    r15 = fmaf(r46, r15, r22 * r25);
    r25 = r4 * r3;
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r20 = r13 + r20;
    r14 = r47 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r20 = fmaf(r14, r1, r19 * r44);
    r47 = r17 * r6;
    r39 = r27 + r39;
    r39 = r39 + r41;
    r39 = fmaf(r10, r39, r9 * r21);
    r47 = r47 * r39;
    r10 = r18 * r7;
    r10 = r10 * r7;
    r10 = r10 * r19;
    r10 = fmaf(r28, r10, r8 * r47);
    r47 = r18 * r6;
    r47 = r47 * r6;
    r47 = r47 * r19;
    r10 = fmaf(r28, r47, r10);
    r21 = r17 * r7;
    r21 = r21 * r14;
    r10 = fmaf(r8, r21, r10);
    r21 = r0 * r10;
    r21 = r21 * r26;
    r20 = fmaf(r33, r21, r20);
    r21 = r4 * r2;
    r47 = r5 * r10;
    r47 = r47 * r26;
    r39 = fmaf(r39, r1, r29 * r47);
    r47 = r19 * r29;
    r39 = fmaf(r35, r47, r39);
    r21 = fmaf(r39, r21, r20 * r25);
    r25 = r4 * r2;
    r47 = r28 * r29;
    r8 = r17 * r47;
    r14 = r5 * r6;
    r8 = fmaf(r14, r8, r1);
    r9 = r18 * r3;
    r41 = r33 * r47;
    r9 = fmaf(r41, r9, r8 * r25);
    WriteSum4<float, float>((float *)inout_shared, r30, r15, r21, r9);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r4 * r3;
    r21 = r18 * r7;
    r21 = r21 * r7;
    r15 = r18 * r6;
    r15 = r15 * r6;
    r15 = fmaf(r28, r15, r28 * r21);
    r21 = r0 * r15;
    r21 = r21 * r26;
    r21 = fmaf(r33, r21, r44);
    r44 = r4 * r2;
    r30 = r5 * r15;
    r30 = r30 * r26;
    r35 = fmaf(r29, r35, r29 * r30);
    r44 = fmaf(r35, r44, r21 * r9);
    r9 = r4 * r3;
    r30 = r0 * r17;
    r30 = r30 * r7;
    r30 = r30 * r28;
    r30 = fmaf(r33, r30, r1);
    r1 = r18 * r2;
    r1 = fmaf(r41, r1, r30 * r9);
    WriteSum2<float, float>((float *)inout_shared, r1, r44);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r48, r48, r32 * r32);
    r1 = fmaf(r46, r46, r22 * r22);
    r9 = fmaf(r39, r39, r20 * r20);
    r28 = r0 * r7;
    r26 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r28 = r28 * r26;
    r28 = r28 * r36;
    r28 = r28 * r29;
    r28 = r28 * r33;
    r28 = r28 * r14;
    r14 = fmaf(r8, r8, r28);
    WriteSum4<float, float>((float *)inout_shared, r44, r1, r9, r14);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = fmaf(r30, r30, r28);
    r14 = fmaf(r21, r21, r35 * r35);
    WriteSum2<float, float>((float *)inout_shared, r28, r14);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r14 = fmaf(r32, r46, r48 * r22);
    r28 = fmaf(r48, r39, r32 * r20);
    r41 = r17 * r41;
    r9 = fmaf(r32, r41, r48 * r8);
    r1 = fmaf(r48, r41, r32 * r30);
    WriteSum4<float, float>((float *)inout_shared, r14, r28, r9, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r32, r21, r48 * r35);
    r48 = fmaf(r46, r20, r22 * r39);
    r1 = fmaf(r46, r41, r22 * r8);
    r9 = fmaf(r22, r41, r46 * r30);
    WriteSum4<float, float>((float *)inout_shared, r32, r48, r1, r9);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r22, r35, r46 * r21);
    r46 = fmaf(r39, r35, r20 * r21);
    r9 = fmaf(r20, r41, r39 * r8);
    r39 = fmaf(r39, r41, r20 * r30);
    WriteSum4<float, float>((float *)inout_shared, r22, r9, r39, r46);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r30, r41, r8 * r41);
    r8 = fmaf(r21, r41, r8 * r35);
    r41 = fmaf(r35, r41, r30 * r21);
    WriteSum3<float, float>((float *)inout_shared, r46, r8, r41);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacFirst(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *pixel, unsigned int pixel_num_alloc, float *focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr,
    float *const out_pose_njtr, unsigned int out_pose_njtr_num_alloc,
    float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, pixel, pixel_num_alloc,
      focal_and_distortion, focal_and_distortion_num_alloc, principal_point,
      principal_point_num_alloc, point, point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc, problem_size);
}

} // namespace caspar