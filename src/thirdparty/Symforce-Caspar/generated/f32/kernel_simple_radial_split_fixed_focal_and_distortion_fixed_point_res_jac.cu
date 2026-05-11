#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_split_fixed_focal_and_distortion_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *focal_and_distortion,
        unsigned int focal_and_distortion_num_alloc, float *point,
        unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *out_pose_jac,
        unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        float *out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
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

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47;
  LoadShared<2, float, float>(principal_point, 0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r25 = 1.0 / r28;
    r39 = r35 * r25;
    r2 = fmaf(r29, r39, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r0 * r25;
    r1 = r1 * r35;
    r3 = fmaf(r7, r1, r3);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r2, r3);
    r39 = r12 * r13;
    r39 = r39 * r18;
    r30 = r30 + r39;
    r24 = fmaf(r10, r24, r11 * r30);
    r30 = r17 * r7;
    r40 = r14 * r14;
    r15 = r15 * r15;
    r41 = r4 * r15;
    r42 = r40 + r41;
    r43 = r4 * r27;
    r44 = r34 + r43;
    r45 = r42 + r44;
    r45 = fmaf(r11, r45, r10 * r33);
    r30 = r30 * r45;
    r33 = r18 * r7;
    r13 = r13 * r19;
    r32 = r32 + r13;
    r43 = r40 + r43;
    r40 = r4 * r34;
    r46 = r15 + r40;
    r43 = r43 + r46;
    r43 = fmaf(r10, r43, r11 * r32);
    r36 = r28 * r36;
    r28 = 1.0 / r36;
    r33 = r33 * r7;
    r33 = r33 * r43;
    r33 = fmaf(r28, r33, r8 * r30);
    r30 = r18 * r6;
    r30 = r30 * r6;
    r30 = r30 * r43;
    r33 = fmaf(r28, r30, r33);
    r32 = r17 * r6;
    r32 = r32 * r24;
    r33 = fmaf(r8, r32, r33);
    r32 = r5 * r33;
    r32 = r32 * r25;
    r32 = fmaf(r29, r32, r24 * r1);
    r24 = r43 * r29;
    r30 = r4 * r35;
    r30 = r30 * r8;
    r32 = fmaf(r30, r24, r32);
    r24 = r0 * r33;
    r47 = r5 * r7;
    r24 = r24 * r25;
    r24 = fmaf(r47, r24, r45 * r1);
    r45 = r0 * r4;
    r45 = r45 * r7;
    r45 = r45 * r8;
    r45 = r45 * r35;
    r24 = fmaf(r43, r45, r24);
    r19 = r12 * r19;
    r38 = r38 + r19;
    r14 = r14 * r14;
    r14 = r14 * r4;
    r15 = r15 + r14;
    r15 = r15 + r44;
    r15 = fmaf(r11, r15, r9 * r38);
    r41 = r34 + r41;
    r14 = r27 + r14;
    r41 = r41 + r14;
    r41 = fmaf(r9, r41, r11 * r22);
    r22 = r41 * r29;
    r22 = fmaf(r30, r22, r15 * r1);
    r34 = r17 * r7;
    r13 = r37 + r13;
    r13 = fmaf(r9, r13, r11 * r16);
    r34 = r34 * r13;
    r16 = r17 * r6;
    r16 = r16 * r15;
    r16 = fmaf(r8, r16, r8 * r34);
    r34 = r18 * r6;
    r34 = r34 * r6;
    r34 = r34 * r41;
    r16 = fmaf(r28, r34, r16);
    r15 = r18 * r7;
    r15 = r15 * r7;
    r15 = r15 * r41;
    r16 = fmaf(r28, r15, r16);
    r15 = r5 * r16;
    r15 = r15 * r25;
    r22 = fmaf(r29, r15, r22);
    r13 = fmaf(r13, r1, r41 * r45);
    r15 = r0 * r16;
    r15 = r15 * r25;
    r13 = fmaf(r47, r15, r13);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r32, r24,
        r22, r13);
    r15 = r0 * r5;
    r15 = r15 * r17;
    r15 = r15 * r6;
    r15 = r15 * r7;
    r15 = r15 * r28;
    r34 = r17 * r6;
    r40 = r27 + r40;
    r40 = r40 + r42;
    r40 = fmaf(r10, r40, r9 * r21);
    r34 = r34 * r40;
    r21 = r18 * r7;
    r19 = r23 + r19;
    r19 = fmaf(r10, r19, r9 * r31);
    r21 = r21 * r7;
    r21 = r21 * r19;
    r21 = fmaf(r28, r21, r8 * r34);
    r34 = r18 * r6;
    r34 = r34 * r6;
    r34 = r34 * r19;
    r21 = fmaf(r28, r34, r21);
    r31 = r17 * r7;
    r20 = r39 + r20;
    r14 = r46 + r14;
    r14 = fmaf(r9, r14, r10 * r20);
    r31 = r31 * r14;
    r21 = fmaf(r8, r31, r21);
    r31 = r5 * r21;
    r31 = r31 * r25;
    r40 = fmaf(r40, r1, r29 * r31);
    r31 = r19 * r29;
    r40 = fmaf(r30, r31, r40);
    r14 = fmaf(r14, r1, r19 * r45);
    r31 = r0 * r21;
    r31 = r31 * r25;
    r14 = fmaf(r47, r31, r14);
    r31 = r28 * r29;
    r34 = r17 * r31;
    r8 = r5 * r6;
    r34 = fmaf(r8, r34, r1);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r40, r14,
        r34, r15);
    r9 = r0 * r17;
    r9 = r9 * r7;
    r9 = r9 * r28;
    r9 = fmaf(r47, r9, r1);
    r1 = r18 * r7;
    r1 = r1 * r7;
    r20 = r18 * r6;
    r20 = r20 * r6;
    r20 = fmaf(r28, r20, r28 * r1);
    r1 = r5 * r20;
    r1 = r1 * r25;
    r30 = fmaf(r29, r30, r29 * r1);
    r1 = r0 * r20;
    r1 = r1 * r25;
    r1 = fmaf(r47, r1, r45);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r15, r9, r30, r1);
    r45 = r4 * r3;
    r25 = r4 * r2;
    r25 = fmaf(r32, r25, r24 * r45);
    r45 = r4 * r2;
    r28 = r4 * r3;
    r28 = fmaf(r13, r28, r22 * r45);
    r45 = r4 * r3;
    r10 = r4 * r2;
    r10 = fmaf(r40, r10, r14 * r45);
    r45 = r4 * r2;
    r46 = r18 * r3;
    r39 = r47 * r31;
    r46 = fmaf(r39, r46, r34 * r45);
    WriteSum4<float, float>((float *)inout_shared, r25, r28, r10, r46);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = r4 * r3;
    r10 = r4 * r2;
    r10 = fmaf(r30, r10, r1 * r46);
    r46 = r4 * r3;
    r28 = r18 * r2;
    r28 = fmaf(r39, r28, r9 * r46);
    WriteSum2<float, float>((float *)inout_shared, r28, r10);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r32, r32, r24 * r24);
    r28 = fmaf(r13, r13, r22 * r22);
    r46 = fmaf(r40, r40, r14 * r14);
    r39 = r0 * r7;
    r25 = 4.00000000000000000e+00;
    r36 = r36 * r36;
    r36 = 1.0 / r36;
    r39 = r39 * r25;
    r39 = r39 * r36;
    r39 = r39 * r29;
    r39 = r39 * r47;
    r39 = r39 * r8;
    r8 = fmaf(r34, r34, r39);
    WriteSum4<float, float>((float *)inout_shared, r10, r28, r46, r8);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = fmaf(r9, r9, r39);
    r8 = fmaf(r1, r1, r30 * r30);
    WriteSum2<float, float>((float *)inout_shared, r39, r8);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r24, r13, r32 * r22);
    r39 = fmaf(r32, r40, r24 * r14);
    r46 = fmaf(r24, r15, r32 * r34);
    r28 = fmaf(r32, r15, r24 * r9);
    WriteSum4<float, float>((float *)inout_shared, r8, r39, r46, r28);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r24 = fmaf(r24, r1, r32 * r30);
    r32 = fmaf(r13, r14, r22 * r40);
    r28 = fmaf(r13, r15, r22 * r34);
    r46 = fmaf(r22, r15, r13 * r9);
    WriteSum4<float, float>((float *)inout_shared, r24, r32, r28, r46);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = fmaf(r22, r30, r13 * r1);
    r13 = fmaf(r40, r30, r14 * r1);
    r46 = fmaf(r14, r15, r40 * r34);
    r40 = fmaf(r40, r15, r14 * r9);
    WriteSum4<float, float>((float *)inout_shared, r22, r46, r40, r13);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r13 = fmaf(r9, r15, r34 * r15);
    r34 = fmaf(r1, r15, r34 * r30);
    r15 = fmaf(r30, r15, r9 * r1);
    WriteSum3<float, float>((float *)inout_shared, r13, r34, r15);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r15 = r4 * r2;
    r34 = r4 * r3;
    WriteSum2<float, float>((float *)inout_shared, r15, r34);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float *)inout_shared, r26, r26);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
}

void SimpleRadialSplitFixedFocalAndDistortionFixedPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *focal_and_distortion,
    unsigned int focal_and_distortion_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    float *out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndDistortionFixedPointResJacKernel<<<n_blocks,
                                                                   1024>>>(
      pose, pose_num_alloc, pose_indices, principal_point,
      principal_point_num_alloc, principal_point_indices, pixel,
      pixel_num_alloc, focal_and_distortion, focal_and_distortion_num_alloc,
      point, point_num_alloc, out_res, out_res_num_alloc, out_pose_jac,
      out_pose_jac_num_alloc, out_pose_njtr, out_pose_njtr_num_alloc,
      out_pose_precond_diag, out_pose_precond_diag_num_alloc,
      out_pose_precond_tril, out_pose_precond_tril_num_alloc,
      out_principal_point_jac, out_principal_point_jac_num_alloc,
      out_principal_point_njtr, out_principal_point_njtr_num_alloc,
      out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar