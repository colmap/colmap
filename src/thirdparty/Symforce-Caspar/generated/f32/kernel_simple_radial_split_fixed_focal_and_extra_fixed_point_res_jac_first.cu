#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
        SharedIndex* principal_point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* point,
        unsigned int point_num_alloc,
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
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r0,
                                         r5,
                                         r6);
    ReadIdx3<1024, float, float, float4>(
        point, 0 * point_num_alloc, global_thread_idx, r7, r8, r9);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r10,
                       r11,
                       r12,
                       r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r14,
                                         r15,
                                         r16,
                                         r17);
    r18 = fmaf(r11, r14, r12 * r17);
    r19 = r10 * r15;
    r18 = fmaf(r4, r19, r18);
    r18 = fmaf(r13, r16, r18);
    r19 = r18 * r18;
    r20 = -2.00000000000000000e+00;
    r19 = r19 * r20;
    r21 = 1.00000000000000000e+00;
    r22 = r13 * r15;
    r23 = fmaf(r11, r17, r22);
    r24 = r10 * r16;
    r25 = r12 * r14;
    r23 = r23 + r24;
    r23 = fmaf(r4, r25, r23);
    r26 = r20 * r23;
    r26 = fmaf(r23, r26, r21);
    r27 = r19 + r26;
    r27 = fmaf(r7, r27, r0);
    r0 = 2.00000000000000000e+00;
    r28 = fmaf(r13, r14, r10 * r17);
    r29 = r11 * r16;
    r28 = fmaf(r4, r29, r28);
    r28 = fmaf(r12, r15, r28);
    r29 = r0 * r28;
    r29 = r29 * r23;
    r30 = r18 * r20;
    r31 = fmaf(r11, r15, r10 * r14);
    r31 = fmaf(r12, r16, r31);
    r31 = fmaf(r4, r31, r13 * r17);
    r30 = fmaf(r31, r30, r29);
    r32 = r0 * r18;
    r32 = r32 * r28;
    r33 = r0 * r31;
    r34 = fmaf(r23, r33, r32);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r35,
                       r36,
                       r37);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r38 = r14 * r16;
    r38 = r38 * r0;
    r39 = r15 * r17;
    r39 = fmaf(r0, r39, r38);
    r40 = r16 * r17;
    r41 = r14 * r15;
    r41 = r41 * r0;
    r40 = fmaf(r20, r40, r41);
    r42 = r15 * r15;
    r42 = r42 * r20;
    r43 = r21 + r42;
    r44 = r16 * r16;
    r44 = r44 * r20;
    r43 = r43 + r44;
    r27 = fmaf(r8, r30, r27);
    r27 = fmaf(r9, r34, r27);
    r27 = fmaf(r37, r39, r27);
    r27 = fmaf(r36, r40, r27);
    r27 = fmaf(r35, r43, r27);
    r34 = 9.99999999999999955e-07;
    r30 = r20 * r23;
    r30 = fmaf(r31, r30, r32);
    r30 = fmaf(r7, r30, r6);
    r6 = r15 * r17;
    r6 = fmaf(r20, r6, r38);
    r42 = r21 + r42;
    r38 = r14 * r14;
    r38 = r38 * r20;
    r42 = r42 + r38;
    r32 = r15 * r16;
    r32 = r32 * r0;
    r45 = r14 * r17;
    r45 = fmaf(r0, r45, r32);
    r46 = r0 * r18;
    r46 = r46 * r23;
    r47 = fmaf(r28, r33, r46);
    r48 = r28 * r28;
    r48 = r48 * r20;
    r26 = r48 + r26;
    r30 = fmaf(r35, r6, r30);
    r30 = fmaf(r37, r42, r30);
    r30 = fmaf(r36, r45, r30);
    r30 = fmaf(r8, r47, r30);
    r30 = fmaf(r9, r26, r30);
    r26 = copysign(1.0, r30);
    r26 = fmaf(r34, r26, r30);
    r34 = 1.0 / r26;
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r30,
                                         r47);
    r49 = r26 * r26;
    r50 = 1.0 / r49;
    r51 = r27 * r50;
    r29 = fmaf(r18, r33, r29);
    r29 = fmaf(r7, r29, r5);
    r5 = r16 * r17;
    r5 = fmaf(r0, r5, r41);
    r44 = r21 + r44;
    r44 = r44 + r38;
    r38 = r14 * r17;
    r38 = fmaf(r20, r38, r32);
    r32 = r28 * r20;
    r32 = fmaf(r31, r32, r46);
    r19 = r21 + r19;
    r19 = r19 + r48;
    r29 = fmaf(r35, r5, r29);
    r29 = fmaf(r36, r44, r29);
    r29 = fmaf(r37, r38, r29);
    r29 = fmaf(r9, r32, r29);
    r29 = fmaf(r8, r19, r29);
    r19 = r29 * r29;
    r32 = fmaf(r50, r19, r27 * r51);
    r32 = fmaf(r47, r32, r21);
    r32 = r30 * r32;
    r37 = r34 * r32;
    r2 = fmaf(r27, r37, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r29, r37, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = fmaf(r2, r2, r3 * r3);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r1);
  if (global_thread_idx < problem_size) {
    r1 = r0 * r23;
    r36 = -5.00000000000000000e-01;
    r35 = r11 * r36;
    r48 = 5.00000000000000000e-01;
    r46 = fmaf(r48, r25, r17 * r35);
    r46 = fmaf(r36, r22, r46);
    r46 = fmaf(r36, r24, r46);
    r1 = r1 * r46;
    r41 = r0 * r18;
    r52 = r11 * r14;
    r53 = r10 * r15;
    r53 = fmaf(r36, r53, r48 * r52);
    r52 = r13 * r16;
    r53 = fmaf(r48, r52, r53);
    r54 = r17 * r48;
    r53 = fmaf(r12, r54, r53);
    r41 = fmaf(r53, r41, r1);
    r52 = r0 * r28;
    r55 = r10 * r17;
    r56 = r13 * r14;
    r56 = fmaf(r36, r56, r36 * r55);
    r55 = r12 * r15;
    r56 = fmaf(r36, r55, r56);
    r57 = r11 * r16;
    r56 = fmaf(r48, r57, r56);
    r52 = r52 * r56;
    r57 = r10 * r14;
    r55 = r12 * r16;
    r55 = fmaf(r36, r55, r36 * r57);
    r55 = fmaf(r13, r54, r55);
    r55 = fmaf(r15, r35, r55);
    r57 = r55 * r33;
    r58 = r52 + r57;
    r59 = r41 + r58;
    r60 = r20 * r31;
    r61 = r23 * r56;
    r60 = fmaf(r20, r61, r53 * r60);
    r62 = r0 * r28;
    r63 = r0 * r18;
    r63 = r63 * r55;
    r62 = fmaf(r46, r62, r63);
    r60 = r60 + r62;
    r60 = fmaf(r7, r60, r8 * r59);
    r59 = r23 * r53;
    r64 = -4.00000000000000000e+00;
    r59 = r59 * r64;
    r65 = r28 * r64;
    r66 = r55 * r65;
    r67 = r59 + r66;
    r60 = fmaf(r9, r67, r60);
    r67 = r4 * r32;
    r67 = r67 * r51;
    r68 = r30 * r47;
    r69 = r0 * r29;
    r70 = r28 * r20;
    r71 = r20 * r31;
    r71 = r71 * r55;
    r70 = fmaf(r56, r70, r71);
    r70 = r70 + r41;
    r41 = r18 * r46;
    r72 = r64 * r41;
    r66 = r66 + r72;
    r66 = fmaf(r8, r66, r9 * r70);
    r70 = r0 * r28;
    r70 = r70 * r53;
    r73 = fmaf(r46, r33, r70);
    r74 = r0 * r23;
    r74 = r74 * r55;
    r75 = r0 * r18;
    r75 = fmaf(r56, r75, r74);
    r73 = r73 + r75;
    r66 = fmaf(r7, r73, r66);
    r69 = r69 * r66;
    r53 = fmaf(r53, r33, r0 * r61);
    r53 = r53 + r62;
    r70 = r74 + r70;
    r74 = r18 * r20;
    r70 = fmaf(r56, r74, r70);
    r73 = r20 * r31;
    r70 = fmaf(r46, r73, r70);
    r70 = fmaf(r8, r70, r9 * r53);
    r72 = r59 + r72;
    r70 = fmaf(r7, r72, r70);
    r72 = r0 * r70;
    r72 = fmaf(r51, r72, r50 * r69);
    r49 = r26 * r49;
    r49 = 1.0 / r49;
    r49 = r20 * r49;
    r26 = r60 * r49;
    r72 = fmaf(r19, r26, r72);
    r69 = r27 * r27;
    r69 = r69 * r49;
    r72 = fmaf(r60, r69, r72);
    r68 = r68 * r72;
    r68 = r68 * r34;
    r72 = fmaf(r27, r68, r60 * r67);
    r72 = fmaf(r70, r37, r72);
    r26 = r4 * r29;
    r26 = r26 * r60;
    r26 = r26 * r50;
    r68 = fmaf(r29, r68, r32 * r26);
    r68 = fmaf(r66, r37, r68);
    r66 = r30 * r47;
    r57 = r1 + r57;
    r1 = r0 * r18;
    r26 = r12 * r17;
    r59 = r10 * r15;
    r59 = fmaf(r48, r59, r36 * r26);
    r26 = r13 * r16;
    r59 = fmaf(r36, r26, r59);
    r59 = fmaf(r14, r35, r59);
    r1 = r1 * r59;
    r26 = r0 * r28;
    r53 = r13 * r14;
    r73 = r12 * r15;
    r73 = fmaf(r48, r73, r48 * r53);
    r73 = fmaf(r10, r54, r73);
    r73 = fmaf(r16, r35, r73);
    r26 = fmaf(r73, r26, r1);
    r57 = r57 + r26;
    r35 = r23 * r55;
    r35 = r35 * r64;
    r53 = r18 * r64;
    r53 = r53 * r73;
    r74 = r35 + r53;
    r74 = fmaf(r7, r74, r9 * r57);
    r57 = r20 * r31;
    r57 = fmaf(r20, r41, r73 * r57);
    r76 = r0 * r28;
    r76 = r76 * r55;
    r77 = r0 * r23;
    r77 = fmaf(r59, r77, r76);
    r57 = r57 + r77;
    r74 = fmaf(r8, r57, r74);
    r57 = r0 * r74;
    r78 = r20 * r23;
    r78 = fmaf(r46, r78, r71);
    r78 = r78 + r26;
    r26 = r0 * r23;
    r26 = r26 * r73;
    r79 = fmaf(r59, r33, r26);
    r79 = r79 + r62;
    r79 = fmaf(r8, r79, r7 * r78);
    r78 = r59 * r65;
    r35 = r35 + r78;
    r79 = fmaf(r9, r35, r79);
    r57 = fmaf(r79, r69, r51 * r57);
    r35 = r79 * r49;
    r57 = fmaf(r19, r35, r57);
    r62 = r0 * r29;
    r26 = r63 + r26;
    r63 = r28 * r20;
    r26 = fmaf(r46, r63, r26);
    r46 = r20 * r31;
    r26 = fmaf(r59, r46, r26);
    r73 = fmaf(r73, r33, r0 * r41);
    r73 = r73 + r77;
    r73 = fmaf(r7, r73, r9 * r26);
    r78 = r53 + r78;
    r73 = fmaf(r8, r78, r73);
    r62 = r62 * r73;
    r57 = fmaf(r50, r62, r57);
    r66 = r66 * r27;
    r66 = r66 * r57;
    r66 = fmaf(r74, r37, r34 * r66);
    r66 = fmaf(r79, r67, r66);
    r62 = r4 * r29;
    r62 = r62 * r79;
    r62 = r62 * r50;
    r62 = fmaf(r32, r62, r73 * r37);
    r73 = r30 * r47;
    r73 = r73 * r29;
    r73 = r73 * r57;
    r62 = fmaf(r34, r73, r62);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r72,
                                          r68,
                                          r66,
                                          r62);
    r25 = fmaf(r36, r25, r11 * r54);
    r25 = fmaf(r48, r22, r25);
    r25 = fmaf(r48, r24, r25);
    r65 = r25 * r65;
    r61 = r64 * r61;
    r24 = r65 + r61;
    r48 = r0 * r18;
    r48 = r48 * r25;
    r76 = r76 + r48;
    r22 = r20 * r23;
    r76 = fmaf(r59, r22, r76);
    r36 = r20 * r31;
    r76 = fmaf(r56, r36, r76);
    r76 = fmaf(r7, r76, r9 * r24);
    r24 = r0 * r28;
    r24 = fmaf(r25, r33, r59 * r24);
    r24 = r24 + r75;
    r76 = fmaf(r8, r24, r76);
    r71 = r52 + r71;
    r52 = r0 * r23;
    r52 = r52 * r25;
    r24 = r18 * r20;
    r71 = fmaf(r59, r24, r71);
    r71 = r71 + r52;
    r55 = r18 * r55;
    r55 = r55 * r64;
    r61 = r55 + r61;
    r61 = fmaf(r7, r61, r8 * r71);
    r33 = fmaf(r56, r33, r48);
    r33 = r33 + r77;
    r61 = fmaf(r9, r33, r61);
    r33 = fmaf(r61, r37, r76 * r67);
    r77 = r30 * r47;
    r56 = r0 * r61;
    r56 = fmaf(r51, r56, r76 * r69);
    r48 = r76 * r49;
    r56 = fmaf(r19, r48, r56);
    r71 = r0 * r29;
    r52 = r1 + r52;
    r52 = r52 + r58;
    r58 = r28 * r20;
    r1 = r20 * r31;
    r1 = fmaf(r25, r1, r59 * r58);
    r1 = r1 + r75;
    r1 = fmaf(r9, r1, r7 * r52);
    r65 = r55 + r65;
    r1 = fmaf(r8, r65, r1);
    r71 = r71 * r1;
    r56 = fmaf(r50, r71, r56);
    r77 = r77 * r27;
    r77 = r77 * r56;
    r33 = fmaf(r34, r77, r33);
    r77 = r30 * r47;
    r77 = r77 * r29;
    r77 = r77 * r56;
    r56 = r4 * r29;
    r56 = r56 * r76;
    r56 = r56 * r50;
    r56 = fmaf(r32, r56, r34 * r77);
    r56 = fmaf(r1, r37, r56);
    r1 = r30 * r47;
    r77 = r6 * r49;
    r71 = r0 * r5;
    r71 = r71 * r29;
    r71 = fmaf(r50, r71, r19 * r77);
    r77 = r0 * r43;
    r71 = fmaf(r51, r77, r71);
    r71 = fmaf(r6, r69, r71);
    r1 = r1 * r27;
    r1 = r1 * r71;
    r1 = fmaf(r34, r1, r43 * r37);
    r1 = fmaf(r6, r67, r1);
    r77 = r4 * r6;
    r77 = r77 * r29;
    r77 = r77 * r50;
    r77 = fmaf(r5, r37, r32 * r77);
    r48 = r30 * r47;
    r48 = r48 * r29;
    r48 = r48 * r71;
    r77 = fmaf(r34, r48, r77);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r33,
                                          r56,
                                          r1,
                                          r77);
    r48 = r30 * r47;
    r71 = r45 * r49;
    r65 = r0 * r44;
    r65 = r65 * r29;
    r65 = fmaf(r50, r65, r19 * r71);
    r71 = r0 * r40;
    r65 = fmaf(r51, r71, r65);
    r65 = fmaf(r45, r69, r65);
    r48 = r48 * r27;
    r48 = r48 * r65;
    r48 = fmaf(r34, r48, r45 * r67);
    r48 = fmaf(r40, r37, r48);
    r71 = r30 * r47;
    r71 = r71 * r29;
    r71 = r71 * r65;
    r71 = fmaf(r44, r37, r34 * r71);
    r65 = r4 * r45;
    r65 = r65 * r29;
    r65 = r65 * r50;
    r71 = fmaf(r32, r65, r71);
    r67 = fmaf(r39, r37, r42 * r67);
    r65 = r30 * r47;
    r8 = r0 * r38;
    r8 = r8 * r29;
    r55 = r42 * r49;
    r55 = fmaf(r19, r55, r50 * r8);
    r8 = r0 * r39;
    r55 = fmaf(r51, r8, r55);
    r55 = fmaf(r42, r69, r55);
    r65 = r65 * r27;
    r65 = r65 * r55;
    r67 = fmaf(r34, r65, r67);
    r65 = r30 * r47;
    r65 = r65 * r29;
    r65 = r65 * r55;
    r65 = fmaf(r34, r65, r38 * r37);
    r37 = r4 * r42;
    r37 = r37 * r29;
    r37 = r37 * r50;
    r65 = fmaf(r32, r37, r65);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r48,
                                          r71,
                                          r67,
                                          r65);
    r37 = r4 * r3;
    r2 = r4 * r2;
    r37 = fmaf(r72, r2, r68 * r37);
    r32 = r4 * r3;
    r32 = fmaf(r66, r2, r62 * r32);
    r50 = r4 * r3;
    r50 = fmaf(r33, r2, r56 * r50);
    r34 = r4 * r3;
    r34 = fmaf(r1, r2, r77 * r34);
    WriteSum4<float, float>((float*)inout_shared, r37, r32, r50, r34);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r4 * r3;
    r34 = fmaf(r48, r2, r71 * r34);
    r50 = r4 * r3;
    r50 = fmaf(r67, r2, r65 * r50);
    WriteSum2<float, float>((float*)inout_shared, r34, r50);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = fmaf(r72, r72, r68 * r68);
    r34 = fmaf(r66, r66, r62 * r62);
    r32 = fmaf(r56, r56, r33 * r33);
    r37 = fmaf(r1, r1, r77 * r77);
    WriteSum4<float, float>((float*)inout_shared, r50, r34, r32, r37);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fmaf(r48, r48, r71 * r71);
    r32 = fmaf(r67, r67, r65 * r65);
    WriteSum2<float, float>((float*)inout_shared, r37, r32);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fmaf(r68, r62, r72 * r66);
    r37 = fmaf(r68, r56, r72 * r33);
    r34 = fmaf(r72, r1, r68 * r77);
    r50 = fmaf(r72, r48, r68 * r71);
    WriteSum4<float, float>((float*)inout_shared, r32, r37, r34, r50);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r68, r65, r72 * r67);
    r72 = fmaf(r62, r56, r66 * r33);
    r50 = fmaf(r66, r1, r62 * r77);
    r34 = fmaf(r66, r48, r62 * r71);
    WriteSum4<float, float>((float*)inout_shared, r68, r72, r50, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r66 = fmaf(r66, r67, r62 * r65);
    r62 = fmaf(r33, r1, r56 * r77);
    r34 = fmaf(r33, r48, r56 * r71);
    r33 = fmaf(r33, r67, r56 * r65);
    WriteSum4<float, float>((float*)inout_shared, r66, r62, r34, r33);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r1, r48, r77 * r71);
    r77 = fmaf(r77, r65, r1 * r67);
    r65 = fmaf(r71, r65, r48 * r67);
    WriteSum3<float, float>((float*)inout_shared, r33, r77, r65);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = r4 * r3;
    WriteSum2<float, float>((float*)inout_shared, r2, r65);
  };
  FlushSumShared<2, float>(out_principal_point_njtr,
                           0 * out_principal_point_njtr_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    WriteSum2<float, float>((float*)inout_shared, r21, r21);
  };
  FlushSumShared<2, float>(out_principal_point_precond_diag,
                           0 * out_principal_point_precond_diag_num_alloc,
                           principal_point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
    SharedIndex* principal_point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* point,
    unsigned int point_num_alloc,
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
    size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialSplitFixedFocalAndExtraFixedPointResJacFirstKernel<<<n_blocks,
                                                                   1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      principal_point,
      principal_point_num_alloc,
      principal_point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      point,
      point_num_alloc,
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
      problem_size);
}

}  // namespace caspar