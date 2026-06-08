#include "kernel_pinhole_split_fixed_focal_fixed_principal_point_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
        float* const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float* const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float* const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc,
        size_t problem_size) {
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r1);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r2, r3);
    r4 = -1.00000000000000000e+00;
    r2 = fmaf(r2, r4, r0);
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r8, r9, r10);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r11,
                       r12,
                       r13,
                       r14);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r15,
                                         r16,
                                         r17,
                                         r18);
    r19 = fmaf(r12, r15, r13 * r18);
    r20 = r11 * r16;
    r19 = fmaf(r4, r20, r19);
    r19 = fmaf(r14, r17, r19);
    r20 = 2.00000000000000000e+00;
    r21 = fmaf(r14, r15, r11 * r18);
    r22 = r12 * r17;
    r21 = fmaf(r4, r22, r21);
    r21 = fmaf(r13, r16, r21);
    r22 = r20 * r21;
    r23 = r19 * r22;
    r24 = -2.00000000000000000e+00;
    r25 = fmaf(r12, r16, r11 * r15);
    r25 = fmaf(r13, r17, r25);
    r25 = fmaf(r4, r25, r14 * r18);
    r26 = r24 * r25;
    r27 = fmaf(r14, r16, r12 * r18);
    r28 = r11 * r17;
    r29 = r13 * r15;
    r27 = r27 + r28;
    r27 = fmaf(r4, r29, r27);
    r30 = fmaf(r27, r26, r23);
    r30 = fmaf(r8, r30, r7);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r7,
                       r31,
                       r32);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r33 = r15 * r17;
    r33 = r33 * r20;
    r34 = r16 * r18;
    r35 = fmaf(r24, r34, r33);
    r36 = r15 * r15;
    r36 = r36 * r24;
    r37 = 1.00000000000000000e+00;
    r38 = r16 * r16;
    r38 = fmaf(r24, r38, r37);
    r39 = r36 + r38;
    r40 = r16 * r17;
    r40 = r40 * r20;
    r41 = r15 * r18;
    r41 = fmaf(r20, r41, r40);
    r42 = r20 * r19;
    r42 = r42 * r27;
    r43 = fmaf(r25, r22, r42);
    r44 = r24 * r27;
    r44 = r44 * r27;
    r45 = r37 + r44;
    r46 = r21 * r21;
    r46 = r46 * r24;
    r45 = r45 + r46;
    r30 = fmaf(r7, r35, r30);
    r30 = fmaf(r32, r39, r30);
    r30 = fmaf(r31, r41, r30);
    r30 = fmaf(r9, r43, r30);
    r30 = fmaf(r10, r45, r30);
    r45 = copysign(1.0, r30);
    r45 = fmaf(r0, r45, r30);
    r0 = 1.0 / r45;
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r30, r43);
    r44 = r37 + r44;
    r47 = r19 * r19;
    r47 = r47 * r24;
    r44 = r44 + r47;
    r44 = fmaf(r8, r44, r5);
    r5 = r27 * r22;
    r48 = fmaf(r19, r26, r5);
    r49 = r20 * r27;
    r49 = fmaf(r25, r49, r23);
    r34 = fmaf(r20, r34, r33);
    r33 = r17 * r18;
    r23 = r15 * r16;
    r23 = r23 * r20;
    r33 = fmaf(r24, r33, r23);
    r50 = r17 * r17;
    r50 = r50 * r24;
    r38 = r50 + r38;
    r44 = fmaf(r9, r48, r44);
    r44 = fmaf(r10, r49, r44);
    r44 = fmaf(r32, r34, r44);
    r44 = fmaf(r31, r33, r44);
    r44 = fmaf(r7, r38, r44);
    r44 = r30 * r44;
    r2 = fmaf(r0, r44, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r20 * r19;
    r1 = fmaf(r25, r1, r5);
    r1 = fmaf(r8, r1, r6);
    r6 = r17 * r18;
    r6 = fmaf(r20, r6, r23);
    r50 = r37 + r50;
    r50 = r50 + r36;
    r36 = r15 * r18;
    r36 = fmaf(r24, r36, r40);
    r42 = fmaf(r21, r26, r42);
    r47 = r37 + r47;
    r47 = r47 + r46;
    r1 = fmaf(r7, r6, r1);
    r1 = fmaf(r31, r50, r1);
    r1 = fmaf(r32, r36, r1);
    r1 = fmaf(r10, r42, r1);
    r1 = fmaf(r9, r47, r1);
    r1 = r43 * r1;
    r3 = fmaf(r0, r1, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r47 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r47);
  if (global_thread_idx < problem_size) {
    r47 = r4 * r2;
    r42 = r20 * r25;
    r32 = r13 * r18;
    r31 = 5.00000000000000000e-01;
    r7 = r12 * r15;
    r7 = fmaf(r31, r7, r31 * r32);
    r32 = r11 * r16;
    r46 = -5.00000000000000000e-01;
    r7 = fmaf(r46, r32, r7);
    r37 = r14 * r31;
    r7 = fmaf(r17, r37, r7);
    r32 = r11 * r18;
    r40 = r14 * r15;
    r40 = fmaf(r46, r40, r46 * r32);
    r32 = r13 * r16;
    r40 = fmaf(r46, r32, r40);
    r23 = r12 * r17;
    r40 = fmaf(r31, r23, r40);
    r23 = r27 * r40;
    r42 = fmaf(r20, r23, r7 * r42);
    r32 = r20 * r19;
    r5 = r11 * r15;
    r49 = r13 * r17;
    r49 = fmaf(r46, r49, r46 * r5);
    r5 = r12 * r46;
    r49 = fmaf(r18, r37, r49);
    r49 = fmaf(r16, r5, r49);
    r32 = r32 * r49;
    r48 = r14 * r16;
    r48 = fmaf(r18, r5, r46 * r48);
    r48 = fmaf(r31, r29, r48);
    r48 = fmaf(r46, r28, r48);
    r51 = fmaf(r48, r22, r32);
    r42 = r42 + r51;
    r52 = r20 * r27;
    r52 = r52 * r49;
    r53 = r19 * r24;
    r53 = fmaf(r40, r53, r52);
    r54 = r7 * r22;
    r53 = r53 + r54;
    r53 = fmaf(r48, r26, r53);
    r53 = fmaf(r9, r53, r10 * r42);
    r42 = r27 * r7;
    r55 = -4.00000000000000000e+00;
    r42 = r42 * r55;
    r56 = r19 * r48;
    r57 = r55 * r56;
    r58 = r42 + r57;
    r53 = fmaf(r8, r58, r53);
    r58 = r30 * r53;
    r59 = r20 * r27;
    r59 = r59 * r48;
    r60 = r20 * r19;
    r60 = fmaf(r7, r60, r59);
    r61 = r20 * r25;
    r61 = r61 * r49;
    r62 = r40 * r22;
    r63 = r61 + r62;
    r64 = r60 + r63;
    r7 = fmaf(r7, r26, r24 * r23);
    r7 = r7 + r51;
    r7 = fmaf(r8, r7, r9 * r64);
    r64 = r49 * r55;
    r65 = r21 * r64;
    r42 = r42 + r65;
    r7 = fmaf(r10, r42, r7);
    r45 = r45 * r45;
    r45 = 1.0 / r45;
    r45 = r4 * r45;
    r44 = r45 * r44;
    r58 = fmaf(r7, r44, r0 * r58);
    r42 = r4 * r3;
    r66 = r21 * r24;
    r67 = r49 * r26;
    r66 = fmaf(r40, r66, r67);
    r66 = r66 + r60;
    r65 = r57 + r65;
    r65 = fmaf(r9, r65, r10 * r66);
    r66 = r20 * r25;
    r66 = fmaf(r48, r66, r54);
    r54 = r20 * r19;
    r54 = fmaf(r40, r54, r52);
    r66 = r66 + r54;
    r65 = fmaf(r8, r66, r65);
    r66 = r43 * r65;
    r52 = r7 * r45;
    r52 = fmaf(r1, r52, r0 * r66);
    r42 = fmaf(r52, r42, r58 * r47);
    r47 = r4 * r3;
    r66 = r24 * r27;
    r66 = fmaf(r48, r66, r67);
    r57 = r20 * r19;
    r60 = r13 * r18;
    r68 = r11 * r16;
    r68 = fmaf(r31, r68, r46 * r60);
    r60 = r14 * r17;
    r68 = fmaf(r46, r60, r68);
    r68 = fmaf(r15, r5, r68);
    r57 = r57 * r68;
    r60 = r11 * r18;
    r69 = r13 * r16;
    r69 = fmaf(r31, r69, r31 * r60);
    r69 = fmaf(r15, r37, r69);
    r69 = fmaf(r17, r5, r69);
    r5 = fmaf(r69, r22, r57);
    r66 = r66 + r5;
    r60 = r20 * r27;
    r60 = r60 * r69;
    r70 = r20 * r25;
    r70 = fmaf(r68, r70, r60);
    r70 = r70 + r51;
    r70 = fmaf(r9, r70, r8 * r66);
    r66 = r21 * r55;
    r66 = r66 * r68;
    r51 = r27 * r64;
    r71 = r66 + r51;
    r70 = fmaf(r10, r71, r70);
    r71 = r70 * r45;
    r60 = r32 + r60;
    r32 = r21 * r24;
    r60 = fmaf(r48, r32, r60);
    r60 = fmaf(r68, r26, r60);
    r32 = r20 * r25;
    r32 = fmaf(r20, r56, r69 * r32);
    r48 = r20 * r27;
    r49 = r49 * r22;
    r48 = fmaf(r68, r48, r49);
    r32 = r32 + r48;
    r32 = fmaf(r8, r32, r10 * r60);
    r60 = r19 * r55;
    r60 = r60 * r69;
    r66 = r66 + r60;
    r32 = fmaf(r9, r66, r32);
    r66 = r43 * r32;
    r66 = fmaf(r0, r66, r1 * r71);
    r71 = r4 * r2;
    r61 = r59 + r61;
    r61 = r61 + r5;
    r51 = r60 + r51;
    r51 = fmaf(r8, r51, r10 * r61);
    r69 = fmaf(r69, r26, r24 * r56);
    r69 = r69 + r48;
    r51 = fmaf(r9, r69, r51);
    r69 = r30 * r51;
    r69 = fmaf(r70, r44, r0 * r69);
    r71 = fmaf(r69, r71, r66 * r47);
    r47 = r4 * r3;
    r56 = r21 * r55;
    r61 = r12 * r18;
    r29 = fmaf(r46, r29, r31 * r61);
    r29 = fmaf(r16, r37, r29);
    r29 = fmaf(r31, r28, r29);
    r56 = r56 * r29;
    r23 = r55 * r23;
    r55 = r56 + r23;
    r28 = r20 * r19;
    r28 = r28 * r29;
    r31 = r24 * r27;
    r31 = fmaf(r68, r31, r28);
    r31 = r31 + r49;
    r31 = fmaf(r40, r26, r31);
    r31 = fmaf(r8, r31, r10 * r55);
    r55 = r20 * r25;
    r22 = fmaf(r68, r22, r29 * r55);
    r22 = r22 + r54;
    r31 = fmaf(r9, r22, r31);
    r22 = r31 * r45;
    r55 = r20 * r27;
    r55 = r55 * r29;
    r57 = r57 + r55;
    r57 = r57 + r63;
    r63 = r21 * r24;
    r26 = fmaf(r29, r26, r68 * r63);
    r26 = r26 + r54;
    r26 = fmaf(r10, r26, r8 * r57);
    r64 = r19 * r64;
    r56 = r56 + r64;
    r26 = fmaf(r9, r56, r26);
    r56 = r43 * r26;
    r56 = fmaf(r0, r56, r1 * r22);
    r22 = r4 * r2;
    r57 = r19 * r24;
    r57 = fmaf(r68, r57, r55);
    r57 = r57 + r62;
    r57 = r57 + r67;
    r64 = r23 + r64;
    r64 = fmaf(r8, r64, r9 * r57);
    r8 = r20 * r25;
    r8 = fmaf(r40, r8, r28);
    r8 = r8 + r48;
    r64 = fmaf(r10, r8, r64);
    r8 = r30 * r64;
    r8 = fmaf(r31, r44, r0 * r8);
    r22 = fmaf(r8, r22, r56 * r47);
    r47 = r4 * r3;
    r10 = r43 * r6;
    r48 = r35 * r45;
    r48 = fmaf(r1, r48, r0 * r10);
    r10 = r4 * r2;
    r28 = r30 * r38;
    r28 = fmaf(r35, r44, r0 * r28);
    r10 = fmaf(r28, r10, r48 * r47);
    WriteSum4<float, float>((float*)inout_shared, r42, r71, r22, r10);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r4 * r3;
    r22 = r41 * r45;
    r71 = r43 * r50;
    r71 = fmaf(r0, r71, r1 * r22);
    r22 = r4 * r2;
    r42 = r30 * r33;
    r42 = fmaf(r0, r42, r41 * r44);
    r22 = fmaf(r42, r22, r71 * r10);
    r10 = r4 * r3;
    r47 = r43 * r36;
    r40 = r39 * r45;
    r40 = fmaf(r1, r40, r0 * r47);
    r47 = r4 * r2;
    r1 = r30 * r34;
    r44 = fmaf(r39, r44, r0 * r1);
    r47 = fmaf(r44, r47, r40 * r10);
    WriteSum2<float, float>((float*)inout_shared, r22, r47);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r52, r52, r58 * r58);
    r22 = fmaf(r66, r66, r69 * r69);
    r10 = fmaf(r8, r8, r56 * r56);
    r1 = fmaf(r28, r28, r48 * r48);
    WriteSum4<float, float>((float*)inout_shared, r47, r22, r10, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r42, r42, r71 * r71);
    r10 = fmaf(r44, r44, r40 * r40);
    WriteSum2<float, float>((float*)inout_shared, r1, r10);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = fmaf(r58, r69, r52 * r66);
    r1 = fmaf(r52, r56, r58 * r8);
    r22 = fmaf(r58, r28, r52 * r48);
    r47 = fmaf(r52, r71, r58 * r42);
    WriteSum4<float, float>((float*)inout_shared, r10, r1, r22, r47);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r52, r40, r58 * r44);
    r58 = fmaf(r69, r8, r66 * r56);
    r47 = fmaf(r66, r48, r69 * r28);
    r22 = fmaf(r66, r71, r69 * r42);
    WriteSum4<float, float>((float*)inout_shared, r52, r58, r47, r22);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fmaf(r69, r44, r66 * r40);
    r66 = fmaf(r8, r28, r56 * r48);
    r22 = fmaf(r56, r71, r8 * r42);
    r8 = fmaf(r8, r44, r56 * r40);
    WriteSum4<float, float>((float*)inout_shared, r69, r66, r22, r8);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r48, r71, r28 * r42);
    r28 = fmaf(r28, r44, r48 * r40);
    r44 = fmaf(r42, r44, r71 * r40);
    WriteSum3<float, float>((float*)inout_shared, r8, r28, r44);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
    float* const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc,
    float* const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float* const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc,
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  PinholeSplitFixedFocalFixedPrincipalPointFixedPointResJacFirstKernel<<<
      n_blocks,
      1024>>>(pose,
              pose_num_alloc,
              pose_indices,
              sensor_from_rig,
              sensor_from_rig_num_alloc,
              pixel,
              pixel_num_alloc,
              focal,
              focal_num_alloc,
              principal_point,
              principal_point_num_alloc,
              point,
              point_num_alloc,
              out_res,
              out_res_num_alloc,
              out_rTr,
              out_pose_njtr,
              out_pose_njtr_num_alloc,
              out_pose_precond_diag,
              out_pose_precond_diag_num_alloc,
              out_pose_precond_tril,
              out_pose_precond_tril_num_alloc,
              problem_size);
}

}  // namespace caspar