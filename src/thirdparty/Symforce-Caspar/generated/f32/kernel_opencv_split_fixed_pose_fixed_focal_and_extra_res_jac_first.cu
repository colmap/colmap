#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_pose_fixed_focal_and_extra_res_jac_first.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedPoseFixedFocalAndExtraResJacFirstKernel(
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *principal_point, unsigned int principal_point_num_alloc,
        SharedIndex *principal_point_indices, float *point,
        unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
        unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
        float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
        float *out_res, unsigned int out_res_num_alloc, float *const out_rTr,
        float *out_principal_point_jac,
        unsigned int out_principal_point_jac_num_alloc,
        float *const out_principal_point_njtr,
        unsigned int out_principal_point_njtr_num_alloc,
        float *const out_principal_point_precond_diag,
        unsigned int out_principal_point_precond_diag_num_alloc,
        float *const out_principal_point_precond_tril,
        unsigned int out_principal_point_precond_tril_num_alloc,
        float *out_point_jac, unsigned int out_point_jac_num_alloc,
        float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
        float *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59;
  LoadShared<2, float, float>(principal_point, 0 * principal_point_num_alloc,
                              principal_point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float *)inout_shared,
                       principal_point_indices_loc[threadIdx.x].target, r0, r1);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx, r2, r3, r4, r5);
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         4 * focal_and_extra_num_alloc,
                                         global_thread_idx, r6, r7);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r8, r9, r10);
  };
  LoadShared<3, float, float>(point, 0 * point_num_alloc, point_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       point_indices_loc[threadIdx.x].target, r11, r12, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
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
    r9 = fmaf(r11, r28, r9);
    ReadIdx3<1024, float, float, float4>(pose, 4 * pose_num_alloc,
                                         global_thread_idx, r22, r30, r31);
    r32 = r15 * r16;
    r32 = r32 * r14;
    r33 = r17 * r18;
    r33 = fmaf(r14, r33, r32);
    r34 = -2.00000000000000000e+00;
    r35 = r15 * r15;
    r35 = r34 * r35;
    r36 = 1.00000000000000000e+00;
    r37 = r17 * r17;
    r37 = fmaf(r34, r37, r36);
    r38 = r35 + r37;
    r39 = r16 * r17;
    r39 = r39 * r14;
    r40 = r15 * r18;
    r40 = fmaf(r34, r40, r39);
    r41 = r14 * r27;
    r41 = r41 * r26;
    r42 = r23 * r29;
    r43 = fmaf(r34, r42, r41);
    r44 = r27 * r27;
    r44 = r34 * r44;
    r45 = r36 + r44;
    r46 = r23 * r23;
    r46 = r46 * r34;
    r45 = r45 + r46;
    r9 = fmaf(r22, r33, r9);
    r9 = fmaf(r30, r38, r9);
    r9 = fmaf(r31, r40, r9);
    r9 = fmaf(r13, r43, r9);
    r9 = fmaf(r12, r45, r9);
    r40 = r9 * r9;
    r38 = 9.99999999999999955e-07;
    r33 = r14 * r27;
    r33 = r33 * r23;
    r23 = r34 * r26;
    r23 = fmaf(r29, r23, r33);
    r10 = fmaf(r11, r23, r10);
    r47 = r15 * r17;
    r47 = r47 * r14;
    r48 = r16 * r18;
    r49 = fmaf(r34, r48, r47);
    r50 = r16 * r16;
    r50 = r50 * r34;
    r51 = r36 + r50;
    r51 = r51 + r35;
    r35 = r15 * r18;
    r35 = fmaf(r14, r35, r39);
    r42 = fmaf(r14, r42, r41);
    r41 = r26 * r26;
    r41 = r34 * r41;
    r39 = r36 + r41;
    r39 = r39 + r46;
    r10 = fmaf(r22, r49, r10);
    r10 = fmaf(r31, r51, r10);
    r10 = fmaf(r30, r35, r10);
    r10 = fmaf(r12, r42, r10);
    r10 = fmaf(r13, r39, r10);
    r35 = copysign(1.0, r10);
    r35 = fmaf(r38, r35, r10);
    r38 = r35 * r35;
    r10 = 1.0 / r38;
    r40 = r40 * r10;
    r41 = r36 + r41;
    r41 = r41 + r44;
    r11 = fmaf(r11, r41, r8);
    r8 = r27 * r34;
    r8 = fmaf(r29, r8, r24);
    r24 = r14 * r26;
    r24 = fmaf(r29, r24, r33);
    r48 = fmaf(r14, r48, r47);
    r47 = r17 * r18;
    r47 = fmaf(r34, r47, r32);
    r37 = r50 + r37;
    r11 = fmaf(r12, r8, r11);
    r11 = fmaf(r13, r24, r11);
    r11 = fmaf(r31, r48, r11);
    r11 = fmaf(r30, r47, r11);
    r11 = fmaf(r22, r37, r11);
    r37 = 3.00000000000000000e+00;
    r22 = r11 * r37;
    r47 = r11 * r10;
    r22 = fmaf(r47, r22, r40);
    r30 = 1.0 / r35;
    r22 = fmaf(r11, r30, r7 * r22);
    r48 = r11 * r47;
    r40 = r48 + r40;
    r5 = r5 * r40;
    r40 = fmaf(r40, r5, r4 * r40);
    r31 = r40 * r30;
    r13 = r14 * r47;
    r12 = r6 * r13;
    r22 = fmaf(r11, r31, r22);
    r22 = fmaf(r9, r12, r22);
    r22 = fmaf(r2, r22, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r0, r50);
    r22 = fmaf(r0, r25, r22);
    r0 = r9 * r9;
    r0 = r0 * r37;
    r0 = fmaf(r10, r0, r48);
    r0 = fmaf(r9, r30, r6 * r0);
    r48 = r7 * r9;
    r0 = fmaf(r13, r48, r0);
    r0 = fmaf(r9, r31, r0);
    r0 = fmaf(r3, r0, r1);
    r0 = fmaf(r50, r25, r0);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r22, r0);
    r50 = fmaf(r22, r22, r0 * r0);
  };
  SumStore<float>(out_rTr_local, (float *)inout_shared, 0,
                  global_thread_idx < problem_size, r50);
  if (global_thread_idx < problem_size) {
    r22 = r25 * r22;
    r50 = r25 * r0;
    WriteSum2<float, float>((float *)inout_shared, r22, r50);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float *)inout_shared, r36, r36);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = 6.00000000000000000e+00;
    r50 = r41 * r36;
    r1 = r11 * r11;
    r38 = r35 * r38;
    r38 = 1.0 / r38;
    r35 = -6.00000000000000000e+00;
    r48 = r23 * r35;
    r1 = r1 * r38;
    r1 = fmaf(r48, r1, r47 * r50);
    r50 = r14 * r28;
    r50 = r50 * r9;
    r32 = r34 * r9;
    r33 = r9 * r38;
    r32 = r32 * r33;
    r50 = fmaf(r23, r32, r10 * r50);
    r1 = r1 + r50;
    r1 = fmaf(r41, r30, r7 * r1);
    r29 = r6 * r23;
    r44 = -4.00000000000000000e+00;
    r44 = r11 * r44;
    r44 = r44 * r33;
    r1 = fmaf(r44, r29, r1);
    r51 = r6 * r14;
    r51 = r51 * r41;
    r51 = r51 * r9;
    r1 = fmaf(r10, r51, r1);
    r49 = r25 * r23;
    r1 = fmaf(r47, r49, r1);
    r46 = r34 * r23;
    r46 = r46 * r11;
    r46 = r46 * r11;
    r46 = fmaf(r38, r46, r41 * r13);
    r50 = r50 + r46;
    r5 = r14 * r5;
    r50 = fmaf(r50, r5, r4 * r50);
    r52 = r11 * r50;
    r1 = fmaf(r30, r52, r1);
    r53 = r25 * r23;
    r53 = r53 * r40;
    r1 = fmaf(r47, r53, r1);
    r1 = fmaf(r28, r12, r1);
    r1 = fmaf(r41, r31, r1);
    r53 = r2 * r1;
    r52 = r28 * r9;
    r52 = r52 * r36;
    r49 = r9 * r33;
    r49 = fmaf(r48, r49, r10 * r52);
    r49 = r49 + r46;
    r46 = r7 * r44;
    r49 = fmaf(r23, r46, r6 * r49);
    r52 = r7 * r28;
    r49 = fmaf(r13, r52, r49);
    r48 = r7 * r14;
    r48 = r48 * r41;
    r48 = r48 * r9;
    r49 = fmaf(r10, r48, r49);
    r51 = r25 * r23;
    r51 = r51 * r9;
    r49 = fmaf(r10, r51, r49);
    r29 = r9 * r50;
    r49 = fmaf(r30, r29, r49);
    r54 = r25 * r23;
    r54 = r54 * r9;
    r54 = r54 * r40;
    r49 = fmaf(r10, r54, r49);
    r49 = fmaf(r28, r30, r49);
    r49 = fmaf(r28, r31, r49);
    r54 = r3 * r49;
    r29 = r8 * r36;
    r51 = r42 * r11;
    r51 = r51 * r11;
    r51 = r51 * r35;
    r51 = fmaf(r38, r51, r47 * r29);
    r29 = r14 * r45;
    r29 = r29 * r9;
    r29 = fmaf(r10, r29, r42 * r32);
    r51 = r51 + r29;
    r48 = r6 * r42;
    r48 = fmaf(r44, r48, r7 * r51);
    r51 = r34 * r42;
    r51 = r51 * r11;
    r51 = r51 * r11;
    r51 = fmaf(r38, r51, r8 * r13);
    r29 = r29 + r51;
    r29 = fmaf(r29, r5, r4 * r29);
    r52 = r11 * r29;
    r48 = fmaf(r30, r52, r48);
    r55 = r6 * r14;
    r55 = r55 * r8;
    r55 = r55 * r9;
    r48 = fmaf(r10, r55, r48);
    r56 = r25 * r42;
    r56 = r56 * r40;
    r48 = fmaf(r47, r56, r48);
    r57 = r25 * r42;
    r48 = fmaf(r47, r57, r48);
    r48 = fmaf(r8, r30, r48);
    r48 = fmaf(r8, r31, r48);
    r48 = fmaf(r45, r12, r48);
    r57 = r2 * r48;
    r56 = r42 * r9;
    r56 = r56 * r35;
    r55 = r45 * r9;
    r55 = r55 * r36;
    r55 = fmaf(r10, r55, r33 * r56);
    r55 = r55 + r51;
    r55 = fmaf(r45, r30, r6 * r55);
    r51 = r9 * r29;
    r55 = fmaf(r30, r51, r55);
    r56 = r25 * r42;
    r56 = r56 * r9;
    r55 = fmaf(r10, r56, r55);
    r52 = r25 * r42;
    r52 = r52 * r9;
    r52 = r52 * r40;
    r55 = fmaf(r10, r52, r55);
    r58 = r7 * r14;
    r58 = r58 * r8;
    r58 = r58 * r9;
    r55 = fmaf(r10, r58, r55);
    r59 = r7 * r45;
    r55 = fmaf(r13, r59, r55);
    r55 = fmaf(r42, r46, r55);
    r55 = fmaf(r45, r31, r55);
    r59 = r3 * r55;
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r53, r54,
        r57, r59);
    r59 = r39 * r11;
    r59 = r59 * r11;
    r59 = r59 * r35;
    r57 = r24 * r36;
    r57 = fmaf(r47, r57, r38 * r59);
    r59 = r14 * r43;
    r59 = r59 * r9;
    r32 = fmaf(r39, r32, r10 * r59);
    r57 = r57 + r32;
    r12 = fmaf(r43, r12, r7 * r57);
    r57 = r25 * r39;
    r12 = fmaf(r47, r57, r12);
    r59 = r6 * r39;
    r12 = fmaf(r44, r59, r12);
    r44 = r34 * r39;
    r44 = r44 * r11;
    r44 = r44 * r11;
    r44 = fmaf(r24, r13, r38 * r44);
    r32 = r32 + r44;
    r5 = fmaf(r32, r5, r4 * r32);
    r32 = r11 * r5;
    r12 = fmaf(r30, r32, r12);
    r4 = r6 * r14;
    r4 = r4 * r24;
    r4 = r4 * r9;
    r12 = fmaf(r10, r4, r12);
    r38 = r25 * r39;
    r38 = r38 * r40;
    r12 = fmaf(r47, r38, r12);
    r12 = fmaf(r24, r30, r12);
    r12 = fmaf(r24, r31, r12);
    r38 = r2 * r12;
    r4 = r43 * r9;
    r4 = r4 * r36;
    r32 = r39 * r9;
    r32 = r32 * r35;
    r32 = fmaf(r33, r32, r10 * r4);
    r32 = r32 + r44;
    r44 = r7 * r43;
    r44 = fmaf(r13, r44, r6 * r32);
    r32 = r25 * r39;
    r32 = r32 * r9;
    r44 = fmaf(r10, r32, r44);
    r13 = r25 * r39;
    r13 = r13 * r9;
    r13 = r13 * r40;
    r44 = fmaf(r10, r13, r44);
    r40 = r9 * r5;
    r44 = fmaf(r30, r40, r44);
    r4 = r7 * r14;
    r4 = r4 * r24;
    r4 = r4 * r9;
    r44 = fmaf(r10, r4, r44);
    r44 = fmaf(r43, r30, r44);
    r44 = fmaf(r39, r46, r44);
    r44 = fmaf(r43, r31, r44);
    r31 = r3 * r44;
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx, r38, r31);
    r31 = r3 * r25;
    r31 = r31 * r0;
    r22 = r2 * r22;
    r31 = fmaf(r1, r22, r49 * r31);
    r38 = r3 * r25;
    r38 = r38 * r0;
    r38 = fmaf(r48, r22, r55 * r38);
    r4 = r3 * r25;
    r4 = r4 * r0;
    r22 = fmaf(r12, r22, r44 * r4);
    WriteSum3<float, float>((float *)inout_shared, r31, r38, r22);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r22 = r3 * r3;
    r38 = r49 * r49;
    r2 = r2 * r2;
    r31 = r1 * r2;
    r1 = fmaf(r1, r31, r22 * r38);
    r38 = r48 * r48;
    r4 = r55 * r22;
    r55 = fmaf(r55, r4, r2 * r38);
    r38 = r12 * r12;
    r0 = r44 * r44;
    r0 = fmaf(r22, r0, r2 * r38);
    WriteSum3<float, float>((float *)inout_shared, r1, r55, r0);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r0 = fmaf(r48, r31, r49 * r4);
    r55 = r49 * r44;
    r31 = fmaf(r12, r31, r22 * r55);
    r55 = r48 * r12;
    r4 = fmaf(r44, r4, r2 * r55);
    WriteSum3<float, float>((float *)inout_shared, r0, r31, r4);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void OpencvSplitFixedPoseFixedFocalAndExtraResJacFirst(
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *principal_point, unsigned int principal_point_num_alloc,
    SharedIndex *principal_point_indices, float *point,
    unsigned int point_num_alloc, SharedIndex *point_indices, float *pixel,
    unsigned int pixel_num_alloc, float *pose, unsigned int pose_num_alloc,
    float *focal_and_extra, unsigned int focal_and_extra_num_alloc,
    float *out_res, unsigned int out_res_num_alloc, float *const out_rTr,
    float *out_principal_point_jac,
    unsigned int out_principal_point_jac_num_alloc,
    float *const out_principal_point_njtr,
    unsigned int out_principal_point_njtr_num_alloc,
    float *const out_principal_point_precond_diag,
    unsigned int out_principal_point_precond_diag_num_alloc,
    float *const out_principal_point_precond_tril,
    unsigned int out_principal_point_precond_tril_num_alloc,
    float *out_point_jac, unsigned int out_point_jac_num_alloc,
    float *const out_point_njtr, unsigned int out_point_njtr_num_alloc,
    float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedPoseFixedFocalAndExtraResJacFirstKernel<<<n_blocks, 1024>>>(
      sensor_from_rig, sensor_from_rig_num_alloc, principal_point,
      principal_point_num_alloc, principal_point_indices, point,
      point_num_alloc, point_indices, pixel, pixel_num_alloc, pose,
      pose_num_alloc, focal_and_extra, focal_and_extra_num_alloc, out_res,
      out_res_num_alloc, out_rTr, out_principal_point_jac,
      out_principal_point_jac_num_alloc, out_principal_point_njtr,
      out_principal_point_njtr_num_alloc, out_principal_point_precond_diag,
      out_principal_point_precond_diag_num_alloc,
      out_principal_point_precond_tril,
      out_principal_point_precond_tril_num_alloc, out_point_jac,
      out_point_jac_num_alloc, out_point_njtr, out_point_njtr_num_alloc,
      out_point_precond_diag, out_point_precond_diag_num_alloc,
      out_point_precond_tril, out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar