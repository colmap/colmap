#include "kernel_simple_radial_split_fixed_focal_and_distortion_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* out_pose_jac,
        unsigned int out_pose_jac_num_alloc,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float* out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float* const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float* const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float* const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        float* out_point_jac,
        unsigned int out_point_jac_num_alloc,
        float* const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float* const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float* const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex principal_point_indices_loc[1024];
  principal_point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? principal_point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});
  __shared__ SharedIndex point_indices_loc[1024];
  point_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? point_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53;
  LoadShared<2, float, float>(principal_point,
                              0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target,
                       r0,
                       r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    ReadIdx2<1024, float, float, float2>(focal_and_distortion,
                                         0 * focal_and_distortion_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>(
        (float*)inout_shared, pose_indices_loc[threadIdx.x].target, r6, r7, r8);
  };
  __syncthreads();
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r9,
                       r10,
                       r11);
  };
  __syncthreads();
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r12,
                       r13,
                       r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r14 * r15;
    r17 = -2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r12 * r13;
    r19 = 2.00000000000000000e+00;
    r18 = r18 * r19;
    r20 = r16 + r18;
    r6 = fmaf(r10, r20, r6);
    r21 = r12 * r14;
    r21 = r21 * r19;
    r22 = r13 * r15;
    r22 = r22 * r19;
    r23 = r21 + r22;
    r24 = r14 * r14;
    r25 = r17 * r24;
    r26 = 1.00000000000000000e+00;
    r27 = r13 * r13;
    r28 = fmaf(r17, r27, r26);
    r29 = r25 + r28;
    r6 = fmaf(r11, r23, r6);
    r6 = fmaf(r9, r29, r6);
    r30 = r0 * r6;
    r31 = r14 * r15;
    r31 = r31 * r19;
    r18 = r18 + r31;
    r7 = fmaf(r9, r18, r7);
    r32 = r13 * r14;
    r32 = r32 * r19;
    r33 = r12 * r15;
    r33 = r33 * r17;
    r34 = r32 + r33;
    r25 = r26 + r25;
    r35 = r12 * r12;
    r36 = r17 * r35;
    r25 = r25 + r36;
    r7 = fmaf(r11, r34, r7);
    r7 = fmaf(r10, r25, r7);
    r37 = r7 * r7;
    r38 = 9.99999999999999955e-07;
    r39 = r12 * r15;
    r39 = r39 * r19;
    r32 = r32 + r39;
    r8 = fmaf(r10, r32, r8);
    r40 = r13 * r15;
    r40 = r40 * r17;
    r21 = r21 + r40;
    r28 = r36 + r28;
    r8 = fmaf(r9, r21, r8);
    r8 = fmaf(r11, r28, r8);
    r36 = copysign(1.0, r8);
    r36 = fmaf(r38, r36, r8);
    r38 = r36 * r36;
    r8 = 1.0 / r38;
    r41 = r6 * r6;
    r41 = fmaf(r8, r41, r8 * r37);
    r41 = fmaf(r5, r41, r26);
    r37 = 1.0 / r36;
    r42 = r41 * r37;
    r2 = fmaf(r30, r42, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r37;
    r1 = r1 * r41;
    r3 = fmaf(r7, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r42 = r12 * r13;
    r42 = r42 * r17;
    r31 = r31 + r42;
    r31 = fmaf(r10, r23, r11 * r31);
    r43 = r19 * r7;
    r15 = r15 * r15;
    r44 = r4 * r15;
    r45 = r24 + r44;
    r46 = r4 * r27;
    r47 = r35 + r46;
    r48 = r45 + r47;
    r48 = fmaf(r11, r48, r10 * r34);
    r43 = r43 * r48;
    r13 = r13 * r14;
    r13 = r13 * r17;
    r33 = r33 + r13;
    r46 = r24 + r46;
    r49 = r4 * r35;
    r50 = r15 + r49;
    r46 = r46 + r50;
    r46 = fmaf(r10, r46, r11 * r33);
    r33 = r7 * r7;
    r33 = r17 * r33;
    r38 = r36 * r38;
    r36 = 1.0 / r38;
    r33 = r33 * r36;
    r43 = fmaf(r46, r33, r8 * r43);
    r51 = r6 * r6;
    r51 = r17 * r51;
    r51 = r51 * r36;
    r52 = r19 * r6;
    r52 = r52 * r31;
    r43 = fmaf(r8, r52, r43);
    r43 = fmaf(r46, r51, r43);
    r52 = r5 * r43;
    r52 = r52 * r37;
    r52 = fmaf(r30, r52, r31 * r1);
    r31 = r46 * r30;
    r41 = r4 * r41;
    r41 = r41 * r8;
    r52 = fmaf(r41, r31, r52);
    r31 = r0 * r43;
    r53 = r5 * r7;
    r31 = r31 * r37;
    r31 = fmaf(r53, r31, r48 * r1);
    r48 = r0 * r7;
    r48 = r48 * r46;
    r31 = fmaf(r41, r48, r31);
    r14 = r12 * r14;
    r14 = r14 * r17;
    r40 = r40 + r14;
    r24 = r4 * r24;
    r15 = r15 + r24;
    r15 = r15 + r47;
    r15 = fmaf(r11, r15, r9 * r40);
    r44 = r35 + r44;
    r24 = r27 + r24;
    r44 = r44 + r24;
    r44 = fmaf(r9, r44, r11 * r21);
    r35 = r44 * r30;
    r35 = fmaf(r41, r35, r15 * r1);
    r40 = r19 * r7;
    r13 = r39 + r13;
    r13 = fmaf(r9, r13, r11 * r18);
    r40 = r40 * r13;
    r11 = r19 * r6;
    r11 = r11 * r15;
    r11 = fmaf(r8, r11, r8 * r40);
    r11 = fmaf(r44, r51, r11);
    r11 = fmaf(r44, r33, r11);
    r40 = r5 * r11;
    r40 = r40 * r37;
    r35 = fmaf(r30, r40, r35);
    r40 = r0 * r7;
    r40 = r40 * r44;
    r13 = fmaf(r13, r1, r41 * r40);
    r40 = r0 * r11;
    r40 = r40 * r37;
    r13 = fmaf(r53, r40, r13);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r52,
                                          r31,
                                          r35,
                                          r13);
    r40 = r0 * r5;
    r40 = r40 * r19;
    r40 = r40 * r6;
    r40 = r40 * r7;
    r40 = r40 * r36;
    r15 = r19 * r6;
    r49 = r27 + r49;
    r49 = r49 + r45;
    r49 = fmaf(r10, r49, r9 * r20);
    r15 = r15 * r49;
    r14 = r22 + r14;
    r14 = fmaf(r10, r14, r9 * r32);
    r15 = fmaf(r14, r33, r8 * r15);
    r22 = r19 * r7;
    r42 = r16 + r42;
    r24 = r50 + r24;
    r24 = fmaf(r9, r24, r10 * r42);
    r22 = r22 * r24;
    r15 = fmaf(r8, r22, r15);
    r15 = fmaf(r14, r51, r15);
    r22 = r5 * r15;
    r22 = r22 * r37;
    r49 = fmaf(r49, r1, r30 * r22);
    r14 = r14 * r41;
    r49 = fmaf(r30, r14, r49);
    r22 = r0 * r7;
    r24 = fmaf(r24, r1, r14 * r22);
    r22 = r0 * r15;
    r22 = r22 * r37;
    r24 = fmaf(r53, r22, r24);
    r22 = r5 * r19;
    r22 = r22 * r6;
    r22 = r22 * r36;
    r22 = fmaf(r30, r22, r1);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r49,
                                          r24,
                                          r22,
                                          r40);
    r14 = r0 * r19;
    r14 = r14 * r7;
    r14 = r14 * r36;
    r14 = fmaf(r53, r14, r1);
    r9 = r33 + r51;
    r42 = r5 * r9;
    r42 = r42 * r37;
    r42 = fmaf(r30, r41, r30 * r42);
    r10 = r0 * r9;
    r10 = r10 * r37;
    r50 = r0 * r7;
    r50 = fmaf(r41, r50, r53 * r10);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r14,
                                          r42,
                                          r50);
    r10 = r4 * r3;
    r16 = r4 * r2;
    r16 = fmaf(r52, r16, r31 * r10);
    r10 = r4 * r2;
    r45 = r4 * r3;
    r45 = fmaf(r13, r45, r35 * r10);
    r10 = r4 * r3;
    r27 = r4 * r2;
    r27 = fmaf(r49, r27, r24 * r10);
    r10 = r4 * r2;
    r36 = r17 * r36;
    r17 = r3 * r36;
    r39 = r30 * r53;
    r17 = fmaf(r39, r17, r22 * r10);
    WriteSum4<float, float>((float*)inout_shared, r16, r45, r27, r17);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r17 = r4 * r3;
    r27 = r4 * r2;
    r27 = fmaf(r42, r27, r50 * r17);
    r17 = r4 * r3;
    r45 = r2 * r36;
    r45 = fmaf(r39, r45, r14 * r17);
    WriteSum2<float, float>((float*)inout_shared, r45, r27);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r27 = fmaf(r52, r52, r31 * r31);
    r45 = fmaf(r13, r13, r35 * r35);
    r17 = fmaf(r49, r49, r24 * r24);
    r16 = r0 * r5;
    r10 = 4.00000000000000000e+00;
    r38 = r38 * r38;
    r38 = 1.0 / r38;
    r16 = r16 * r6;
    r16 = r16 * r7;
    r16 = r16 * r10;
    r16 = r16 * r38;
    r16 = r16 * r39;
    r39 = fmaf(r22, r22, r16);
    WriteSum4<float, float>((float*)inout_shared, r27, r45, r17, r39);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r16 = fmaf(r14, r14, r16);
    r39 = fmaf(r50, r50, r42 * r42);
    WriteSum2<float, float>((float*)inout_shared, r16, r39);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r31, r13, r52 * r35);
    r16 = fmaf(r52, r49, r31 * r24);
    r17 = fmaf(r31, r40, r52 * r22);
    r45 = fmaf(r52, r40, r31 * r14);
    WriteSum4<float, float>((float*)inout_shared, r39, r16, r17, r45);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r31 = fmaf(r31, r50, r52 * r42);
    r52 = fmaf(r13, r24, r35 * r49);
    r45 = fmaf(r13, r40, r35 * r22);
    r17 = fmaf(r35, r40, r13 * r14);
    WriteSum4<float, float>((float*)inout_shared, r31, r52, r45, r17);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r35 = fmaf(r35, r42, r13 * r50);
    r13 = fmaf(r49, r42, r24 * r50);
    r17 = fmaf(r24, r40, r49 * r22);
    r49 = fmaf(r49, r40, r24 * r14);
    WriteSum4<float, float>((float*)inout_shared, r35, r17, r49, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r14, r40, r22 * r40);
    r22 = fmaf(r50, r40, r22 * r42);
    r40 = fmaf(r42, r40, r14 * r50);
    WriteSum3<float, float>((float*)inout_shared, r13, r22, r40);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r40 = r4 * r2;
    r22 = r4 * r3;
    WriteSum2<float, float>((float*)inout_shared, r40, r22);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r26, r26);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = r21 * r30;
    r22 = r19 * r29;
    r22 = r22 * r6;
    r22 = fmaf(r8, r22, r21 * r51);
    r40 = r19 * r18;
    r40 = r40 * r7;
    r22 = fmaf(r8, r40, r22);
    r22 = fmaf(r21, r33, r22);
    r40 = r5 * r22;
    r40 = r40 * r37;
    r40 = fmaf(r30, r40, r41 * r26);
    r40 = fmaf(r29, r1, r40);
    r26 = r0 * r22;
    r26 = r26 * r37;
    r13 = r0 * r21;
    r13 = r13 * r7;
    r13 = fmaf(r41, r13, r53 * r26);
    r13 = fmaf(r18, r1, r13);
    r26 = r32 * r30;
    r26 = fmaf(r20, r1, r41 * r26);
    r42 = r19 * r20;
    r42 = r42 * r6;
    r42 = fmaf(r32, r51, r8 * r42);
    r50 = r19 * r25;
    r50 = r50 * r7;
    r42 = fmaf(r8, r50, r42);
    r42 = fmaf(r32, r33, r42);
    r50 = r5 * r42;
    r50 = r50 * r37;
    r26 = fmaf(r30, r50, r26);
    r50 = r0 * r32;
    r50 = r50 * r7;
    r14 = r0 * r42;
    r14 = r14 * r37;
    r14 = fmaf(r53, r14, r41 * r50);
    r14 = fmaf(r25, r1, r14);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r40,
                                          r13,
                                          r26,
                                          r14);
    r50 = r19 * r23;
    r50 = r50 * r6;
    r51 = fmaf(r28, r51, r8 * r50);
    r50 = r19 * r34;
    r50 = r50 * r7;
    r51 = fmaf(r8, r50, r51);
    r51 = fmaf(r28, r33, r51);
    r33 = r5 * r51;
    r33 = r33 * r37;
    r33 = fmaf(r30, r33, r23 * r1);
    r50 = r28 * r30;
    r33 = fmaf(r41, r50, r33);
    r50 = r0 * r28;
    r50 = r50 * r7;
    r50 = fmaf(r41, r50, r34 * r1);
    r1 = r0 * r51;
    r1 = r1 * r37;
    r50 = fmaf(r53, r1, r50);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r33,
                                          r50);
    r1 = r4 * r2;
    r53 = r4 * r3;
    r53 = fmaf(r13, r53, r40 * r1);
    r1 = r4 * r2;
    r37 = r4 * r3;
    r37 = fmaf(r14, r37, r26 * r1);
    r1 = r4 * r3;
    r41 = r4 * r2;
    r41 = fmaf(r33, r41, r50 * r1);
    WriteSum3<float, float>((float*)inout_shared, r53, r37, r41);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r41 = fmaf(r40, r40, r13 * r13);
    r37 = fmaf(r14, r14, r26 * r26);
    r53 = fmaf(r33, r33, r50 * r50);
    WriteSum3<float, float>((float*)inout_shared, r41, r37, r53);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r53 = fmaf(r13, r14, r40 * r26);
    r13 = fmaf(r13, r50, r40 * r33);
    r50 = fmaf(r14, r50, r26 * r33);
    WriteSum3<float, float>((float*)inout_shared, r53, r13, r50);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndDistortionResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* out_pose_jac,
    unsigned int out_pose_jac_num_alloc,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float* out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float* const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float* const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float* const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    float* out_point_jac,
    unsigned int out_point_jac_num_alloc,
    float* const out_point_njtr,
    unsigned int out_point_njtr_num_alloc,
    float* const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float* const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionResJacKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_distortion,
      focal_and_distortion_num_alloc,
      out_res,
      out_res_num_alloc,
      out_pose_jac,
      out_pose_jac_num_alloc,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      out_principal_point_jac,
      out_principal_point_jac_num_alloc,
      out_principal_point_njtr,
      out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc,
      out_point_jac,
      out_point_jac_num_alloc,
      out_point_njtr,
      out_point_njtr_num_alloc,
      out_point_precond_diag,
      out_point_precond_diag_num_alloc,
      out_point_precond_tril,
      out_point_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar