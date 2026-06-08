#include "kernel_pinhole_split_fixed_focal_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    PinholeSplitFixedFocalResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal,
        unsigned int focal_num_alloc,
        float* out_res,
        unsigned int out_res_num_alloc,
        float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80;
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
    r0 = 9.99999999999999955e-07;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r5,
                                         r6,
                                         r7);
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r8,
                       r9,
                       r10);
  };
  __syncthreads();
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
    r7 = fmaf(r8, r30, r7);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r31,
                       r32,
                       r33);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r34 = r15 * r17;
    r34 = r34 * r20;
    r35 = r16 * r18;
    r36 = fmaf(r24, r35, r34);
    r37 = r15 * r15;
    r37 = r37 * r24;
    r38 = 1.00000000000000000e+00;
    r39 = r16 * r16;
    r39 = fmaf(r24, r39, r38);
    r40 = r37 + r39;
    r41 = r16 * r17;
    r41 = r41 * r20;
    r42 = r15 * r18;
    r42 = fmaf(r20, r42, r41);
    r43 = r20 * r19;
    r43 = r43 * r27;
    r44 = fmaf(r25, r22, r43);
    r45 = r24 * r27;
    r45 = r45 * r27;
    r46 = r38 + r45;
    r47 = r21 * r21;
    r47 = r47 * r24;
    r46 = r46 + r47;
    r7 = fmaf(r31, r36, r7);
    r7 = fmaf(r33, r40, r7);
    r7 = fmaf(r32, r42, r7);
    r7 = fmaf(r9, r44, r7);
    r7 = fmaf(r10, r46, r7);
    r48 = copysign(1.0, r7);
    r48 = fmaf(r0, r48, r7);
    r0 = 1.0 / r48;
    ReadIdx2<1024, float, float, float2>(
        focal, 0 * focal_num_alloc, global_thread_idx, r7, r49);
    r45 = r38 + r45;
    r50 = r19 * r19;
    r50 = r50 * r24;
    r45 = r45 + r50;
    r5 = fmaf(r8, r45, r5);
    r51 = r27 * r22;
    r52 = fmaf(r19, r26, r51);
    r53 = r20 * r27;
    r53 = fmaf(r25, r53, r23);
    r35 = fmaf(r20, r35, r34);
    r34 = r17 * r18;
    r23 = r15 * r16;
    r23 = r23 * r20;
    r34 = fmaf(r24, r34, r23);
    r54 = r17 * r17;
    r54 = r54 * r24;
    r39 = r54 + r39;
    r5 = fmaf(r9, r52, r5);
    r5 = fmaf(r10, r53, r5);
    r5 = fmaf(r33, r35, r5);
    r5 = fmaf(r32, r34, r5);
    r5 = fmaf(r31, r39, r5);
    r5 = r7 * r5;
    r2 = fmaf(r0, r5, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r20 * r19;
    r1 = fmaf(r25, r1, r51);
    r6 = fmaf(r8, r1, r6);
    r51 = r17 * r18;
    r51 = fmaf(r20, r51, r23);
    r54 = r38 + r54;
    r54 = r54 + r37;
    r37 = r15 * r18;
    r37 = fmaf(r24, r37, r41);
    r43 = fmaf(r21, r26, r43);
    r50 = r38 + r50;
    r50 = r50 + r47;
    r6 = fmaf(r31, r51, r6);
    r6 = fmaf(r32, r54, r6);
    r6 = fmaf(r33, r37, r6);
    r6 = fmaf(r10, r43, r6);
    r6 = fmaf(r9, r50, r6);
    r6 = r49 * r6;
    r3 = fmaf(r0, r6, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r33 = fmaf(r3, r3, r2 * r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r33);
  if (global_thread_idx < problem_size) {
    r33 = r20 * r25;
    r32 = r13 * r18;
    r31 = 5.00000000000000000e-01;
    r47 = r12 * r15;
    r47 = fmaf(r31, r47, r31 * r32);
    r32 = r11 * r16;
    r41 = -5.00000000000000000e-01;
    r47 = fmaf(r41, r32, r47);
    r23 = r14 * r31;
    r47 = fmaf(r17, r23, r47);
    r32 = r11 * r18;
    r55 = r14 * r15;
    r55 = fmaf(r41, r55, r41 * r32);
    r32 = r13 * r16;
    r55 = fmaf(r41, r32, r55);
    r56 = r12 * r17;
    r55 = fmaf(r31, r56, r55);
    r56 = r27 * r55;
    r33 = fmaf(r20, r56, r47 * r33);
    r32 = r20 * r19;
    r57 = r11 * r15;
    r58 = r13 * r17;
    r58 = fmaf(r41, r58, r41 * r57);
    r57 = r12 * r41;
    r58 = fmaf(r18, r23, r58);
    r58 = fmaf(r16, r57, r58);
    r32 = r32 * r58;
    r59 = r14 * r16;
    r59 = fmaf(r18, r57, r41 * r59);
    r59 = fmaf(r31, r29, r59);
    r59 = fmaf(r41, r28, r59);
    r60 = fmaf(r59, r22, r32);
    r33 = r33 + r60;
    r61 = r20 * r27;
    r61 = r61 * r58;
    r62 = r19 * r24;
    r62 = fmaf(r55, r62, r61);
    r63 = r47 * r22;
    r62 = r62 + r63;
    r62 = fmaf(r59, r26, r62);
    r62 = fmaf(r9, r62, r10 * r33);
    r33 = r27 * r47;
    r64 = -4.00000000000000000e+00;
    r33 = r33 * r64;
    r65 = r19 * r59;
    r66 = r64 * r65;
    r67 = r33 + r66;
    r62 = fmaf(r8, r67, r62);
    r67 = r7 * r62;
    r68 = r20 * r27;
    r68 = r68 * r59;
    r69 = r20 * r19;
    r69 = fmaf(r47, r69, r68);
    r70 = r20 * r25;
    r70 = r70 * r58;
    r71 = r55 * r22;
    r72 = r70 + r71;
    r73 = r69 + r72;
    r47 = fmaf(r47, r26, r24 * r56);
    r47 = r47 + r60;
    r47 = fmaf(r8, r47, r9 * r73);
    r73 = r58 * r64;
    r74 = r21 * r73;
    r33 = r33 + r74;
    r47 = fmaf(r10, r33, r47);
    r48 = r48 * r48;
    r48 = 1.0 / r48;
    r48 = r4 * r48;
    r5 = r48 * r5;
    r67 = fmaf(r47, r5, r0 * r67);
    r33 = r21 * r24;
    r75 = r58 * r26;
    r33 = fmaf(r55, r33, r75);
    r33 = r33 + r69;
    r74 = r66 + r74;
    r74 = fmaf(r9, r74, r10 * r33);
    r33 = r20 * r25;
    r33 = fmaf(r59, r33, r63);
    r63 = r20 * r19;
    r63 = fmaf(r55, r63, r61);
    r33 = r33 + r63;
    r74 = fmaf(r8, r33, r74);
    r33 = r49 * r74;
    r61 = r47 * r48;
    r61 = fmaf(r6, r61, r0 * r33);
    r70 = r68 + r70;
    r68 = r20 * r19;
    r33 = r13 * r18;
    r66 = r11 * r16;
    r66 = fmaf(r31, r66, r41 * r33);
    r33 = r14 * r17;
    r66 = fmaf(r41, r33, r66);
    r66 = fmaf(r15, r57, r66);
    r68 = r68 * r66;
    r33 = r11 * r18;
    r69 = r13 * r16;
    r69 = fmaf(r31, r69, r31 * r33);
    r69 = fmaf(r15, r23, r69);
    r69 = fmaf(r17, r57, r69);
    r57 = fmaf(r69, r22, r68);
    r70 = r70 + r57;
    r33 = r19 * r64;
    r33 = r33 * r69;
    r76 = r27 * r73;
    r77 = r33 + r76;
    r77 = fmaf(r8, r77, r10 * r70);
    r70 = fmaf(r69, r26, r24 * r65);
    r78 = r20 * r27;
    r58 = r58 * r22;
    r78 = fmaf(r66, r78, r58);
    r70 = r70 + r78;
    r77 = fmaf(r9, r70, r77);
    r70 = r7 * r77;
    r79 = r24 * r27;
    r79 = fmaf(r59, r79, r75);
    r79 = r79 + r57;
    r57 = r20 * r27;
    r57 = r57 * r69;
    r80 = r20 * r25;
    r80 = fmaf(r66, r80, r57);
    r80 = r80 + r60;
    r80 = fmaf(r9, r80, r8 * r79);
    r79 = r21 * r64;
    r79 = r79 * r66;
    r76 = r79 + r76;
    r80 = fmaf(r10, r76, r80);
    r70 = fmaf(r80, r5, r0 * r70);
    r76 = r80 * r48;
    r57 = r32 + r57;
    r32 = r21 * r24;
    r57 = fmaf(r59, r32, r57);
    r57 = fmaf(r66, r26, r57);
    r32 = r20 * r25;
    r65 = fmaf(r20, r65, r69 * r32);
    r65 = r65 + r78;
    r65 = fmaf(r8, r65, r10 * r57);
    r79 = r33 + r79;
    r65 = fmaf(r9, r79, r65);
    r79 = r49 * r65;
    r79 = fmaf(r0, r79, r6 * r76);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r61,
                                          r70,
                                          r79);
    r76 = r20 * r27;
    r33 = r12 * r18;
    r29 = fmaf(r41, r29, r31 * r33);
    r29 = fmaf(r16, r23, r29);
    r29 = fmaf(r31, r28, r29);
    r76 = r76 * r29;
    r28 = r19 * r24;
    r28 = fmaf(r66, r28, r76);
    r28 = r28 + r71;
    r28 = r28 + r75;
    r73 = r19 * r73;
    r56 = r64 * r56;
    r75 = r73 + r56;
    r75 = fmaf(r8, r75, r9 * r28);
    r28 = r20 * r19;
    r28 = r28 * r29;
    r71 = r20 * r25;
    r71 = fmaf(r55, r71, r28);
    r71 = r71 + r78;
    r75 = fmaf(r10, r71, r75);
    r71 = r7 * r75;
    r64 = r21 * r64;
    r64 = r64 * r29;
    r56 = r64 + r56;
    r78 = r24 * r27;
    r78 = fmaf(r66, r78, r28);
    r78 = r78 + r58;
    r78 = fmaf(r55, r26, r78);
    r78 = fmaf(r8, r78, r10 * r56);
    r56 = r20 * r25;
    r22 = fmaf(r66, r22, r29 * r56);
    r22 = r22 + r63;
    r78 = fmaf(r9, r22, r78);
    r71 = fmaf(r78, r5, r0 * r71);
    r22 = r78 * r48;
    r76 = r68 + r76;
    r76 = r76 + r72;
    r72 = r21 * r24;
    r26 = fmaf(r29, r26, r66 * r72);
    r26 = r26 + r63;
    r26 = fmaf(r10, r26, r8 * r76);
    r73 = r64 + r73;
    r26 = fmaf(r9, r73, r26);
    r73 = r49 * r26;
    r73 = fmaf(r0, r73, r6 * r22);
    r22 = r7 * r39;
    r22 = fmaf(r36, r5, r0 * r22);
    r9 = r49 * r51;
    r64 = r36 * r48;
    r64 = fmaf(r6, r64, r0 * r9);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r71,
                                          r73,
                                          r22,
                                          r64);
    r9 = r7 * r34;
    r9 = fmaf(r0, r9, r42 * r5);
    r10 = r42 * r48;
    r76 = r49 * r54;
    r76 = fmaf(r0, r76, r6 * r10);
    r10 = r7 * r35;
    r10 = fmaf(r40, r5, r0 * r10);
    r8 = r49 * r37;
    r63 = r40 * r48;
    r63 = fmaf(r6, r63, r0 * r8);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r76,
                                          r10,
                                          r63);
    r8 = r4 * r2;
    r29 = r4 * r3;
    r29 = fmaf(r61, r29, r67 * r8);
    r8 = r4 * r3;
    r72 = r4 * r2;
    r72 = fmaf(r70, r72, r79 * r8);
    r8 = r4 * r3;
    r66 = r4 * r2;
    r66 = fmaf(r71, r66, r73 * r8);
    r8 = r4 * r3;
    r68 = r4 * r2;
    r68 = fmaf(r22, r68, r64 * r8);
    WriteSum4<float, float>((float*)inout_shared, r29, r72, r66, r68);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r4 * r3;
    r66 = r4 * r2;
    r66 = fmaf(r9, r66, r76 * r68);
    r68 = r4 * r3;
    r72 = r4 * r2;
    r72 = fmaf(r10, r72, r63 * r68);
    WriteSum2<float, float>((float*)inout_shared, r66, r72);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fmaf(r61, r61, r67 * r67);
    r66 = fmaf(r79, r79, r70 * r70);
    r68 = fmaf(r71, r71, r73 * r73);
    r29 = fmaf(r22, r22, r64 * r64);
    WriteSum4<float, float>((float*)inout_shared, r72, r66, r68, r29);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = fmaf(r9, r9, r76 * r76);
    r68 = fmaf(r10, r10, r63 * r63);
    WriteSum2<float, float>((float*)inout_shared, r29, r68);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r67, r70, r61 * r79);
    r29 = fmaf(r61, r73, r67 * r71);
    r66 = fmaf(r67, r22, r61 * r64);
    r72 = fmaf(r61, r76, r67 * r9);
    WriteSum4<float, float>((float*)inout_shared, r68, r29, r66, r72);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r61, r63, r67 * r10);
    r67 = fmaf(r70, r71, r79 * r73);
    r72 = fmaf(r79, r64, r70 * r22);
    r66 = fmaf(r79, r76, r70 * r9);
    WriteSum4<float, float>((float*)inout_shared, r61, r67, r72, r66);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r70 = fmaf(r70, r10, r79 * r63);
    r79 = fmaf(r71, r22, r73 * r64);
    r66 = fmaf(r73, r76, r71 * r9);
    r71 = fmaf(r71, r10, r73 * r63);
    WriteSum4<float, float>((float*)inout_shared, r70, r79, r66, r71);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r71 = fmaf(r64, r76, r22 * r9);
    r22 = fmaf(r22, r10, r64 * r63);
    r10 = fmaf(r9, r10, r76 * r63);
    WriteSum3<float, float>((float*)inout_shared, r71, r22, r10);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r10 = r4 * r2;
    r22 = r4 * r3;
    WriteSum2<float, float>((float*)inout_shared, r10, r22);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r38, r38);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = r7 * r45;
    r38 = fmaf(r0, r38, r30 * r5);
    r22 = r49 * r1;
    r10 = r30 * r48;
    r10 = fmaf(r6, r10, r0 * r22);
    r22 = r7 * r52;
    r22 = fmaf(r0, r22, r44 * r5);
    r71 = r44 * r48;
    r9 = r49 * r50;
    r9 = fmaf(r0, r9, r6 * r71);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r38,
                                          r10,
                                          r22,
                                          r9);
    r71 = r7 * r53;
    r5 = fmaf(r46, r5, r0 * r71);
    r71 = r49 * r43;
    r63 = r46 * r48;
    r63 = fmaf(r6, r63, r0 * r71);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r5, r63);
    r71 = r4 * r3;
    r6 = r4 * r2;
    r6 = fmaf(r38, r6, r10 * r71);
    r71 = r4 * r3;
    r0 = r4 * r2;
    r0 = fmaf(r22, r0, r9 * r71);
    r71 = r4 * r3;
    r76 = r4 * r2;
    r76 = fmaf(r5, r76, r63 * r71);
    WriteSum3<float, float>((float*)inout_shared, r6, r0, r76);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fmaf(r38, r38, r10 * r10);
    r0 = fmaf(r9, r9, r22 * r22);
    r6 = fmaf(r63, r63, r5 * r5);
    WriteSum3<float, float>((float*)inout_shared, r76, r0, r6);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r38, r22, r10 * r9);
    r38 = fmaf(r38, r5, r10 * r63);
    r5 = fmaf(r22, r5, r9 * r63);
    WriteSum3<float, float>((float*)inout_shared, r6, r38, r5);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void PinholeSplitFixedFocalResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal,
    unsigned int focal_num_alloc,
    float* out_res,
    unsigned int out_res_num_alloc,
    float* const out_rTr,
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
  PinholeSplitFixedFocalResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal,
      focal_num_alloc,
      out_res,
      out_res_num_alloc,
      out_rTr,
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