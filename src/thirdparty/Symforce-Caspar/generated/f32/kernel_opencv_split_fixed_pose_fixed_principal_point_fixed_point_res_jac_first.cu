#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_pose_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel(
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        SharedIndex *focal_and_extra_indices, float *pixel,
        unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        float *point, unsigned int point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *const out_rTr,
        float *const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float *const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float *const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
        size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
  };
  LoadShared<4, float, float>(focal_and_extra, 0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target, r2, r3,
                       r4, r5);
  };
  __syncthreads();
  LoadShared<2, float, float>(focal_and_extra, 4 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target, r6, r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r11, r12, r13);
    r14 = 2.00000000000000000e+00;
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r15, r16, r17, r18);
    ReadIdx4<1024, float, float, float4>(pose, 0 * pose_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r15, r22, r18 * r19);
    r24 = r17 * r20;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r16, r21, r23);
    r24 = r14 * r23;
    r26 = r15 * r21;
    r26 = fmaf(r25, r26, r18 * r20);
    r26 = fmaf(r16, r22, r26);
    r26 = fmaf(r17, r19, r26);
    r24 = r24 * r26;
    r27 = fmaf(r15, r20, r18 * r21);
    r28 = r16 * r19;
    r27 = fmaf(r25, r28, r27);
    r27 = fmaf(r17, r22, r27);
    r28 = r14 * r27;
    r29 = fmaf(r16, r20, r15 * r19);
    r29 = fmaf(r17, r21, r29);
    r29 = fmaf(r25, r29, r18 * r22);
    r28 = fmaf(r29, r28, r24);
    r28 = fmaf(r11, r28, r9);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r9, r22, r30);
    r31 = r15 * r16;
    r31 = r31 * r14;
    r32 = r17 * r18;
    r32 = fmaf(r14, r32, r31);
    r33 = -2.00000000000000000e+00;
    r34 = r15 * r15;
    r34 = r33 * r34;
    r35 = 1.00000000000000000e+00;
    r36 = r17 * r17;
    r36 = fmaf(r33, r36, r35);
    r37 = r34 + r36;
    r38 = r16 * r17;
    r38 = r38 * r14;
    r39 = r15 * r18;
    r39 = fmaf(r33, r39, r38);
    r40 = r14 * r27;
    r40 = r40 * r26;
    r41 = r33 * r29;
    r42 = fmaf(r23, r41, r40);
    r43 = r27 * r27;
    r43 = r33 * r43;
    r44 = r35 + r43;
    r45 = r23 * r23;
    r45 = r33 * r45;
    r44 = r44 + r45;
    r28 = fmaf(r9, r32, r28);
    r28 = fmaf(r22, r37, r28);
    r28 = fmaf(r30, r39, r28);
    r28 = fmaf(r13, r42, r28);
    r28 = fmaf(r12, r44, r28);
    r44 = r28 * r28;
    r42 = 9.99999999999999955e-07;
    r39 = r14 * r27;
    r39 = r39 * r23;
    r37 = fmaf(r26, r41, r39);
    r37 = fmaf(r11, r37, r10);
    r10 = r15 * r17;
    r10 = r10 * r14;
    r32 = r16 * r18;
    r46 = fmaf(r33, r32, r10);
    r47 = r16 * r16;
    r47 = r47 * r33;
    r48 = r35 + r47;
    r48 = r48 + r34;
    r34 = r15 * r18;
    r34 = fmaf(r14, r34, r38);
    r38 = r14 * r23;
    r38 = fmaf(r29, r38, r40);
    r40 = r26 * r26;
    r40 = r33 * r40;
    r49 = r35 + r40;
    r49 = r49 + r45;
    r37 = fmaf(r9, r46, r37);
    r37 = fmaf(r30, r48, r37);
    r37 = fmaf(r22, r34, r37);
    r37 = fmaf(r12, r38, r37);
    r37 = fmaf(r13, r49, r37);
    r49 = copysign(1.0, r37);
    r49 = fmaf(r42, r49, r37);
    r42 = r49 * r49;
    r37 = 1.0 / r42;
    r44 = r44 * r37;
    r40 = r35 + r40;
    r40 = r40 + r43;
    r40 = fmaf(r11, r40, r8);
    r41 = fmaf(r27, r41, r24);
    r24 = r14 * r26;
    r24 = fmaf(r29, r24, r39);
    r32 = fmaf(r14, r32, r10);
    r10 = r17 * r18;
    r10 = fmaf(r33, r10, r31);
    r36 = r47 + r36;
    r40 = fmaf(r12, r41, r40);
    r40 = fmaf(r13, r24, r40);
    r40 = fmaf(r30, r32, r40);
    r40 = fmaf(r22, r10, r40);
    r40 = fmaf(r9, r36, r40);
    r36 = r40 * r40;
    r9 = 3.00000000000000000e+00;
    r36 = r36 * r9;
    r36 = fmaf(r37, r36, r44);
    r10 = 1.0 / r49;
    r22 = fmaf(r40, r10, r7 * r36);
    r32 = r40 * r40;
    r32 = r32 * r37;
    r44 = r32 + r44;
    r30 = r44 * r44;
    r5 = fmaf(r5, r30, r4 * r44);
    r4 = r40 * r5;
    r22 = fmaf(r10, r4, r22);
    r24 = r14 * r37;
    r13 = r6 * r24;
    r41 = r40 * r28;
    r22 = fmaf(r41, r13, r22);
    r0 = fmaf(r2, r22, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r13, r4);
    r0 = fmaf(r13, r25, r0);
    r13 = r28 * r28;
    r13 = r13 * r9;
    r13 = fmaf(r37, r13, r32);
    r32 = fmaf(r28, r10, r6 * r13);
    r9 = r7 * r24;
    r32 = fmaf(r41, r9, r32);
    r12 = r28 * r5;
    r32 = fmaf(r10, r12, r32);
    r1 = fmaf(r3, r32, r1);
    r1 = fmaf(r4, r25, r1);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r0, r1);
    r4 = fmaf(r0, r0, r1 * r1);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r4);
  if (global_thread_idx < problem_size) {
    r4 = r25 * r32;
    r4 = r4 * r1;
    r12 = r25 * r0;
    r9 = r22 * r12;
    r47 = r25 * r44;
    r31 = r3 * r28;
    r47 = r47 * r1;
    r47 = r47 * r10;
    r39 = r44 * r10;
    r29 = r2 * r40;
    r39 = r39 * r29;
    r39 = fmaf(r12, r39, r31 * r47);
    r47 = r25 * r1;
    r11 = r10 * r30;
    r47 = r47 * r31;
    r8 = r29 * r11;
    r8 = fmaf(r12, r8, r11 * r47);
    WriteSum4<float, float>((float *)inout_shared, r9, r4, r39, r8);
  };
  FlushSumShared<4, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r3 * r25;
    r8 = r8 * r13;
    r39 = r33 * r28;
    r39 = r39 * r0;
    r39 = r39 * r37;
    r39 = fmaf(r29, r39, r1 * r8);
    r8 = r33 * r40;
    r8 = r8 * r1;
    r8 = r8 * r37;
    r0 = r2 * r36;
    r0 = fmaf(r12, r0, r31 * r8);
    WriteSum2<float, float>((float *)inout_shared, r39, r0);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           4 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r22 * r22;
    r39 = r32 * r32;
    r8 = r28 * r37;
    r12 = r3 * r31;
    r8 = r8 * r30;
    r4 = r40 * r37;
    r9 = r2 * r29;
    r4 = r4 * r30;
    r4 = fmaf(r9, r4, r12 * r8);
    r8 = r28 * r28;
    r8 = r37 * r8;
    r47 = r3 * r3;
    r43 = r44 * r30;
    r8 = r8 * r47;
    r8 = r8 * r43;
    r43 = r40 * r44;
    r43 = r43 * r44;
    r43 = r43 * r37;
    r43 = r43 * r30;
    r43 = fmaf(r9, r43, r44 * r8);
    WriteSum4<float, float>((float *)inout_shared, r0, r39, r4, r43);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = r28 * r9;
    r4 = 4.00000000000000000e+00;
    r42 = r49 * r42;
    r49 = r49 * r42;
    r49 = 1.0 / r49;
    r49 = r4 * r49;
    r49 = r49 * r41;
    r4 = r3 * r3;
    r39 = r13 * r13;
    r4 = fmaf(r39, r4, r49 * r43);
    r43 = r40 * r12;
    r39 = r2 * r2;
    r39 = r39 * r36;
    r39 = fmaf(r36, r39, r49 * r43);
    WriteSum2<float, float>((float *)inout_shared, r4, r39);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           4 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r39 = 0.00000000000000000e+00;
    r4 = r44 * r22;
    r4 = r4 * r10;
    r4 = r4 * r29;
    r43 = r22 * r29;
    r49 = r11 * r43;
    r0 = r28 * r24;
    r0 = r0 * r43;
    WriteSum4<float, float>((float *)inout_shared, r39, r4, r49, r0);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = r2 * r36;
    r0 = r0 * r22;
    r22 = r3 * r13;
    r22 = r22 * r32;
    r49 = r44 * r32;
    r49 = r49 * r10;
    r49 = r49 * r31;
    r31 = r32 * r31;
    r32 = r11 * r31;
    WriteSum4<float, float>((float *)inout_shared, r0, r49, r32, r22);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           4 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r40 * r24;
    r22 = r22 * r31;
    r31 = r40 * r44;
    r31 = r31 * r37;
    r31 = r31 * r30;
    r31 = fmaf(r9, r31, r8);
    r8 = r44 * r9;
    r42 = 1.0 / r42;
    r42 = r14 * r42;
    r42 = r42 * r41;
    r41 = r13 * r10;
    r32 = r44 * r12;
    r41 = fmaf(r32, r41, r42 * r8);
    r8 = r44 * r10;
    r49 = r36 * r9;
    r8 = fmaf(r49, r8, r42 * r32);
    WriteSum4<float, float>((float *)inout_shared, r22, r31, r41, r8);
  };
  FlushSumShared<4, float>(out_focal_and_extra_precond_tril,
                           8 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r30 * r9;
    r41 = r13 * r12;
    r41 = fmaf(r11, r41, r42 * r8);
    r8 = r30 * r12;
    r8 = fmaf(r11, r49, r42 * r8);
    r42 = r40 * r13;
    r42 = r42 * r12;
    r31 = r28 * r24;
    r31 = fmaf(r49, r31, r24 * r42);
    WriteSum3<float, float>((float *)inout_shared, r41, r8, r31);
  };
  FlushSumShared<3, float>(out_focal_and_extra_precond_tril,
                           12 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedPrincipalPointFixedPointResJacFirst(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    SharedIndex *focal_and_extra_indices, float *pixel,
    unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    float *point, unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *const out_rTr,
    float *const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float *const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float *const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
    size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPoseFixedPrincipalPointFixedPointResJacFirstKernel<<<n_blocks,
                                                                       1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, focal_and_extra,
      focal_and_extra_num_alloc, focal_and_extra_indices, pixel,
      pixel_num_alloc, pose, pose_num_alloc, principal_point,
      principal_point_num_alloc, point, point_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc, out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc, problem_size);
}

} // namespace caspar