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
      r76, r77, r78, r79;
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
    r33 = fmaf(r2, r2, r3 * r3);
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
    r33 = r47 * r48;
    r75 = r21 * r24;
    r76 = r58 * r26;
    r75 = fmaf(r55, r75, r76);
    r75 = r75 + r69;
    r74 = r66 + r74;
    r74 = fmaf(r9, r74, r10 * r75);
    r75 = r20 * r25;
    r75 = fmaf(r59, r75, r63);
    r63 = r20 * r19;
    r63 = fmaf(r55, r63, r61);
    r75 = r75 + r63;
    r74 = fmaf(r8, r75, r74);
    r75 = r49 * r74;
    r75 = fmaf(r0, r75, r6 * r33);
    r33 = r24 * r27;
    r33 = fmaf(r59, r33, r76);
    r61 = r20 * r19;
    r66 = r13 * r18;
    r69 = r11 * r16;
    r69 = fmaf(r31, r69, r41 * r66);
    r66 = r14 * r17;
    r69 = fmaf(r41, r66, r69);
    r69 = fmaf(r15, r57, r69);
    r61 = r61 * r69;
    r66 = r11 * r18;
    r77 = r13 * r16;
    r77 = fmaf(r31, r77, r31 * r66);
    r77 = fmaf(r15, r23, r77);
    r77 = fmaf(r17, r57, r77);
    r57 = fmaf(r77, r22, r61);
    r33 = r33 + r57;
    r66 = r20 * r27;
    r66 = r66 * r77;
    r78 = r20 * r25;
    r78 = fmaf(r69, r78, r66);
    r78 = r78 + r60;
    r78 = fmaf(r9, r78, r8 * r33);
    r33 = r21 * r64;
    r33 = r33 * r69;
    r60 = r27 * r73;
    r79 = r33 + r60;
    r78 = fmaf(r10, r79, r78);
    r70 = r68 + r70;
    r70 = r70 + r57;
    r57 = r19 * r64;
    r57 = r57 * r77;
    r60 = r57 + r60;
    r60 = fmaf(r8, r60, r10 * r70);
    r70 = fmaf(r77, r26, r24 * r65);
    r68 = r20 * r27;
    r58 = r58 * r22;
    r68 = fmaf(r69, r68, r58);
    r70 = r70 + r68;
    r60 = fmaf(r9, r70, r60);
    r70 = r7 * r60;
    r70 = fmaf(r0, r70, r78 * r5);
    r79 = r78 * r48;
    r66 = r32 + r66;
    r32 = r21 * r24;
    r66 = fmaf(r59, r32, r66);
    r66 = fmaf(r69, r26, r66);
    r32 = r20 * r25;
    r65 = fmaf(r20, r65, r77 * r32);
    r65 = r65 + r68;
    r65 = fmaf(r8, r65, r10 * r66);
    r57 = r33 + r57;
    r65 = fmaf(r9, r57, r65);
    r57 = r49 * r65;
    r57 = fmaf(r0, r57, r6 * r79);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r67,
                                          r75,
                                          r70,
                                          r57);
    r79 = r21 * r64;
    r33 = r12 * r18;
    r29 = fmaf(r41, r29, r31 * r33);
    r29 = fmaf(r16, r23, r29);
    r29 = fmaf(r31, r28, r29);
    r79 = r79 * r29;
    r56 = r64 * r56;
    r64 = r79 + r56;
    r28 = r20 * r19;
    r28 = r28 * r29;
    r31 = r24 * r27;
    r31 = fmaf(r69, r31, r28);
    r31 = r31 + r58;
    r31 = fmaf(r55, r26, r31);
    r31 = fmaf(r8, r31, r10 * r64);
    r64 = r20 * r25;
    r22 = fmaf(r69, r22, r29 * r64);
    r22 = r22 + r63;
    r31 = fmaf(r9, r22, r31);
    r22 = r20 * r27;
    r22 = r22 * r29;
    r64 = r19 * r24;
    r64 = fmaf(r69, r64, r22);
    r64 = r64 + r71;
    r64 = r64 + r76;
    r73 = r19 * r73;
    r56 = r56 + r73;
    r56 = fmaf(r8, r56, r9 * r64);
    r64 = r20 * r25;
    r64 = fmaf(r55, r64, r28);
    r64 = r64 + r68;
    r56 = fmaf(r10, r64, r56);
    r64 = r7 * r56;
    r64 = fmaf(r0, r64, r31 * r5);
    r22 = r61 + r22;
    r22 = r22 + r72;
    r72 = r21 * r24;
    r26 = fmaf(r29, r26, r69 * r72);
    r26 = r26 + r63;
    r26 = fmaf(r10, r26, r8 * r22);
    r73 = r79 + r73;
    r26 = fmaf(r9, r73, r26);
    r73 = r49 * r26;
    r9 = r31 * r48;
    r9 = fmaf(r6, r9, r0 * r73);
    r73 = r7 * r39;
    r73 = fmaf(r36, r5, r0 * r73);
    r79 = r49 * r51;
    r10 = r36 * r48;
    r10 = fmaf(r6, r10, r0 * r79);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r64,
                                          r9,
                                          r73,
                                          r10);
    r79 = r7 * r34;
    r79 = fmaf(r0, r79, r42 * r5);
    r22 = r42 * r48;
    r8 = r49 * r54;
    r8 = fmaf(r0, r8, r6 * r22);
    r22 = r7 * r35;
    r22 = fmaf(r40, r5, r0 * r22);
    r63 = r49 * r37;
    r29 = r40 * r48;
    r29 = fmaf(r6, r29, r0 * r63);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r79,
                                          r8,
                                          r22,
                                          r29);
    r63 = r4 * r3;
    r72 = r4 * r2;
    r72 = fmaf(r67, r72, r75 * r63);
    r63 = r4 * r2;
    r69 = r4 * r3;
    r69 = fmaf(r57, r69, r70 * r63);
    r63 = r4 * r2;
    r61 = r4 * r3;
    r61 = fmaf(r9, r61, r64 * r63);
    r63 = r4 * r2;
    r68 = r4 * r3;
    r68 = fmaf(r10, r68, r73 * r63);
    WriteSum4<float, float>((float*)inout_shared, r72, r69, r61, r68);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = r4 * r2;
    r61 = r4 * r3;
    r61 = fmaf(r8, r61, r79 * r68);
    r68 = r4 * r3;
    r69 = r4 * r2;
    r69 = fmaf(r22, r69, r29 * r68);
    WriteSum2<float, float>((float*)inout_shared, r61, r69);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fmaf(r67, r67, r75 * r75);
    r61 = fmaf(r57, r57, r70 * r70);
    r68 = fmaf(r9, r9, r64 * r64);
    r72 = fmaf(r10, r10, r73 * r73);
    WriteSum4<float, float>((float*)inout_shared, r69, r61, r68, r72);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r72 = fmaf(r8, r8, r79 * r79);
    r68 = fmaf(r22, r22, r29 * r29);
    WriteSum2<float, float>((float*)inout_shared, r72, r68);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r67, r70, r75 * r57);
    r72 = fmaf(r67, r64, r75 * r9);
    r61 = fmaf(r75, r10, r67 * r73);
    r69 = fmaf(r67, r79, r75 * r8);
    WriteSum4<float, float>((float*)inout_shared, r68, r72, r61, r69);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r75, r29, r67 * r22);
    r67 = fmaf(r70, r64, r57 * r9);
    r69 = fmaf(r57, r10, r70 * r73);
    r61 = fmaf(r70, r79, r57 * r8);
    WriteSum4<float, float>((float*)inout_shared, r75, r67, r69, r61);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = fmaf(r57, r29, r70 * r22);
    r70 = fmaf(r64, r73, r9 * r10);
    r61 = fmaf(r9, r8, r64 * r79);
    r64 = fmaf(r64, r22, r9 * r29);
    WriteSum4<float, float>((float*)inout_shared, r57, r70, r61, r64);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r73, r79, r10 * r8);
    r73 = fmaf(r73, r22, r10 * r29);
    r29 = fmaf(r8, r29, r79 * r22);
    WriteSum3<float, float>((float*)inout_shared, r64, r73, r29);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r4 * r2;
    r73 = r4 * r3;
    WriteSum2<float, float>((float*)inout_shared, r29, r73);
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
    r73 = r49 * r1;
    r29 = r30 * r48;
    r29 = fmaf(r6, r29, r0 * r73);
    r73 = r7 * r52;
    r73 = fmaf(r0, r73, r44 * r5);
    r64 = r49 * r50;
    r8 = r44 * r48;
    r8 = fmaf(r6, r8, r0 * r64);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r38,
                                          r29,
                                          r73,
                                          r8);
    r64 = r7 * r53;
    r5 = fmaf(r46, r5, r0 * r64);
    r64 = r49 * r43;
    r22 = r46 * r48;
    r22 = fmaf(r6, r22, r0 * r64);
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r5, r22);
    r64 = r4 * r2;
    r6 = r4 * r3;
    r6 = fmaf(r29, r6, r38 * r64);
    r64 = r4 * r3;
    r0 = r4 * r2;
    r0 = fmaf(r73, r0, r8 * r64);
    r64 = r4 * r3;
    r79 = r4 * r2;
    r79 = fmaf(r5, r79, r22 * r64);
    WriteSum3<float, float>((float*)inout_shared, r6, r0, r79);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r79 = fmaf(r29, r29, r38 * r38);
    r0 = fmaf(r8, r8, r73 * r73);
    r6 = fmaf(r22, r22, r5 * r5);
    WriteSum3<float, float>((float*)inout_shared, r79, r0, r6);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r6 = fmaf(r38, r73, r29 * r8);
    r29 = fmaf(r29, r22, r38 * r5);
    r5 = fmaf(r73, r5, r8 * r22);
    WriteSum3<float, float>((float*)inout_shared, r6, r29, r5);
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