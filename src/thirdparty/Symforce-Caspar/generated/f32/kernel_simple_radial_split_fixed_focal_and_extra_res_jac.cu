#include "kernel_simple_radial_split_fixed_focal_and_extra_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraResJacKernel(
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
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
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
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88;
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
  };
  LoadShared<3, float, float>(
      point, 0 * point_num_alloc, point_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       point_indices_loc[threadIdx.x].target,
                       r7,
                       r8,
                       r9);
  };
  __syncthreads();
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
    r0 = fmaf(r7, r27, r0);
    r28 = 2.00000000000000000e+00;
    r29 = fmaf(r13, r14, r10 * r17);
    r30 = r11 * r16;
    r29 = fmaf(r4, r30, r29);
    r29 = fmaf(r12, r15, r29);
    r30 = r28 * r29;
    r30 = r30 * r23;
    r31 = r18 * r20;
    r32 = fmaf(r11, r15, r10 * r14);
    r32 = fmaf(r12, r16, r32);
    r32 = fmaf(r4, r32, r13 * r17);
    r31 = fmaf(r32, r31, r30);
    r33 = r28 * r18;
    r33 = r33 * r29;
    r34 = r28 * r32;
    r35 = fmaf(r23, r34, r33);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r36,
                       r37,
                       r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = r14 * r16;
    r39 = r39 * r28;
    r40 = r15 * r17;
    r40 = fmaf(r28, r40, r39);
    r41 = r16 * r17;
    r42 = r14 * r15;
    r42 = r42 * r28;
    r41 = fmaf(r20, r41, r42);
    r43 = r15 * r15;
    r43 = r43 * r20;
    r44 = r21 + r43;
    r45 = r16 * r16;
    r45 = r45 * r20;
    r44 = r44 + r45;
    r0 = fmaf(r8, r31, r0);
    r0 = fmaf(r9, r35, r0);
    r0 = fmaf(r38, r40, r0);
    r0 = fmaf(r37, r41, r0);
    r0 = fmaf(r36, r44, r0);
    r46 = 9.99999999999999955e-07;
    r47 = r20 * r23;
    r47 = fmaf(r32, r47, r33);
    r6 = fmaf(r7, r47, r6);
    r33 = r15 * r17;
    r33 = fmaf(r20, r33, r39);
    r43 = r21 + r43;
    r39 = r14 * r14;
    r39 = r39 * r20;
    r43 = r43 + r39;
    r48 = r15 * r16;
    r48 = r48 * r28;
    r49 = r14 * r17;
    r49 = fmaf(r28, r49, r48);
    r50 = r28 * r18;
    r50 = r50 * r23;
    r51 = fmaf(r29, r34, r50);
    r52 = r29 * r29;
    r52 = r52 * r20;
    r26 = r52 + r26;
    r6 = fmaf(r36, r33, r6);
    r6 = fmaf(r38, r43, r6);
    r6 = fmaf(r37, r49, r6);
    r6 = fmaf(r8, r51, r6);
    r6 = fmaf(r9, r26, r6);
    r53 = copysign(1.0, r6);
    r53 = fmaf(r46, r53, r6);
    r46 = 1.0 / r53;
    ReadIdx2<1024, float, float, float2>(focal_and_extra,
                                         0 * focal_and_extra_num_alloc,
                                         global_thread_idx,
                                         r6,
                                         r54);
    r55 = r53 * r53;
    r56 = 1.0 / r55;
    r57 = r0 * r56;
    r30 = fmaf(r18, r34, r30);
    r5 = fmaf(r7, r30, r5);
    r58 = r16 * r17;
    r58 = fmaf(r28, r58, r42);
    r45 = r21 + r45;
    r45 = r45 + r39;
    r39 = r14 * r17;
    r39 = fmaf(r20, r39, r48);
    r48 = r29 * r20;
    r48 = fmaf(r32, r48, r50);
    r19 = r21 + r19;
    r19 = r19 + r52;
    r5 = fmaf(r36, r58, r5);
    r5 = fmaf(r37, r45, r5);
    r5 = fmaf(r38, r39, r5);
    r5 = fmaf(r9, r48, r5);
    r5 = fmaf(r8, r19, r5);
    r38 = r5 * r5;
    r37 = fmaf(r56, r38, r0 * r57);
    r37 = fmaf(r54, r37, r21);
    r37 = r6 * r37;
    r36 = r46 * r37;
    r2 = fmaf(r0, r36, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r5, r36, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r28 * r23;
    r52 = -5.00000000000000000e-01;
    r50 = r11 * r52;
    r42 = 5.00000000000000000e-01;
    r59 = fmaf(r42, r25, r17 * r50);
    r59 = fmaf(r52, r22, r59);
    r59 = fmaf(r52, r24, r59);
    r1 = r1 * r59;
    r60 = r28 * r18;
    r61 = r11 * r14;
    r62 = r10 * r15;
    r62 = fmaf(r52, r62, r42 * r61);
    r61 = r13 * r16;
    r62 = fmaf(r42, r61, r62);
    r63 = r17 * r42;
    r62 = fmaf(r12, r63, r62);
    r60 = fmaf(r62, r60, r1);
    r61 = r28 * r29;
    r64 = r10 * r17;
    r65 = r13 * r14;
    r65 = fmaf(r52, r65, r52 * r64);
    r64 = r12 * r15;
    r65 = fmaf(r52, r64, r65);
    r66 = r11 * r16;
    r65 = fmaf(r42, r66, r65);
    r61 = r61 * r65;
    r66 = r10 * r14;
    r64 = r12 * r16;
    r64 = fmaf(r52, r64, r52 * r66);
    r64 = fmaf(r13, r63, r64);
    r64 = fmaf(r15, r50, r64);
    r66 = r64 * r34;
    r67 = r61 + r66;
    r68 = r60 + r67;
    r69 = r20 * r32;
    r70 = r23 * r65;
    r69 = fmaf(r20, r70, r62 * r69);
    r71 = r28 * r29;
    r72 = r28 * r18;
    r72 = r72 * r64;
    r71 = fmaf(r59, r71, r72);
    r69 = r69 + r71;
    r69 = fmaf(r7, r69, r8 * r68);
    r68 = r23 * r62;
    r73 = -4.00000000000000000e+00;
    r68 = r68 * r73;
    r74 = r29 * r73;
    r75 = r64 * r74;
    r76 = r68 + r75;
    r69 = fmaf(r9, r76, r69);
    r76 = r4 * r37;
    r76 = r76 * r57;
    r77 = r6 * r54;
    r78 = r28 * r5;
    r79 = r29 * r20;
    r80 = r20 * r32;
    r80 = r80 * r64;
    r79 = fmaf(r65, r79, r80);
    r79 = r79 + r60;
    r60 = r18 * r59;
    r81 = r73 * r60;
    r75 = r75 + r81;
    r75 = fmaf(r8, r75, r9 * r79);
    r79 = r28 * r29;
    r79 = r79 * r62;
    r82 = fmaf(r59, r34, r79);
    r83 = r28 * r23;
    r83 = r83 * r64;
    r84 = r28 * r18;
    r84 = fmaf(r65, r84, r83);
    r82 = r82 + r84;
    r75 = fmaf(r7, r82, r75);
    r78 = r78 * r75;
    r62 = fmaf(r62, r34, r28 * r70);
    r62 = r62 + r71;
    r79 = r83 + r79;
    r83 = r18 * r20;
    r79 = fmaf(r65, r83, r79);
    r82 = r20 * r32;
    r79 = fmaf(r59, r82, r79);
    r79 = fmaf(r8, r79, r9 * r62);
    r81 = r68 + r81;
    r79 = fmaf(r7, r81, r79);
    r81 = r28 * r79;
    r81 = fmaf(r57, r81, r56 * r78);
    r55 = r53 * r55;
    r55 = 1.0 / r55;
    r55 = r20 * r55;
    r53 = r69 * r55;
    r81 = fmaf(r38, r53, r81);
    r78 = r0 * r0;
    r78 = r78 * r55;
    r81 = fmaf(r69, r78, r81);
    r77 = r77 * r81;
    r77 = r77 * r46;
    r81 = fmaf(r0, r77, r69 * r76);
    r81 = fmaf(r79, r36, r81);
    r53 = r4 * r5;
    r53 = r53 * r69;
    r53 = r53 * r56;
    r77 = fmaf(r5, r77, r37 * r53);
    r77 = fmaf(r75, r36, r77);
    r75 = r6 * r54;
    r66 = r1 + r66;
    r1 = r28 * r18;
    r53 = r12 * r17;
    r68 = r10 * r15;
    r68 = fmaf(r42, r68, r52 * r53);
    r53 = r13 * r16;
    r68 = fmaf(r52, r53, r68);
    r68 = fmaf(r14, r50, r68);
    r1 = r1 * r68;
    r53 = r28 * r29;
    r62 = r13 * r14;
    r82 = r12 * r15;
    r82 = fmaf(r42, r82, r42 * r62);
    r82 = fmaf(r10, r63, r82);
    r82 = fmaf(r16, r50, r82);
    r53 = fmaf(r82, r53, r1);
    r66 = r66 + r53;
    r50 = r23 * r64;
    r50 = r50 * r73;
    r62 = r18 * r73;
    r62 = r62 * r82;
    r83 = r50 + r62;
    r83 = fmaf(r7, r83, r9 * r66);
    r66 = r20 * r32;
    r66 = fmaf(r20, r60, r82 * r66);
    r85 = r28 * r29;
    r85 = r85 * r64;
    r86 = r28 * r23;
    r86 = fmaf(r68, r86, r85);
    r66 = r66 + r86;
    r83 = fmaf(r8, r66, r83);
    r66 = r28 * r83;
    r87 = r20 * r23;
    r87 = fmaf(r59, r87, r80);
    r87 = r87 + r53;
    r53 = r28 * r23;
    r53 = r53 * r82;
    r88 = fmaf(r68, r34, r53);
    r88 = r88 + r71;
    r88 = fmaf(r8, r88, r7 * r87);
    r87 = r68 * r74;
    r50 = r50 + r87;
    r88 = fmaf(r9, r50, r88);
    r66 = fmaf(r88, r78, r57 * r66);
    r50 = r88 * r55;
    r66 = fmaf(r38, r50, r66);
    r71 = r28 * r5;
    r53 = r72 + r53;
    r72 = r29 * r20;
    r53 = fmaf(r59, r72, r53);
    r59 = r20 * r32;
    r53 = fmaf(r68, r59, r53);
    r82 = fmaf(r82, r34, r28 * r60);
    r82 = r82 + r86;
    r82 = fmaf(r7, r82, r9 * r53);
    r87 = r62 + r87;
    r82 = fmaf(r8, r87, r82);
    r71 = r71 * r82;
    r66 = fmaf(r56, r71, r66);
    r75 = r75 * r0;
    r75 = r75 * r66;
    r75 = fmaf(r83, r36, r46 * r75);
    r75 = fmaf(r88, r76, r75);
    r71 = r4 * r5;
    r71 = r71 * r88;
    r71 = r71 * r56;
    r71 = fmaf(r37, r71, r82 * r36);
    r82 = r6 * r54;
    r82 = r82 * r5;
    r82 = r82 * r66;
    r71 = fmaf(r46, r82, r71);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r81,
                                          r77,
                                          r75,
                                          r71);
    r25 = fmaf(r52, r25, r11 * r63);
    r25 = fmaf(r42, r22, r25);
    r25 = fmaf(r42, r24, r25);
    r74 = r25 * r74;
    r70 = r73 * r70;
    r24 = r74 + r70;
    r42 = r28 * r18;
    r42 = r42 * r25;
    r85 = r85 + r42;
    r22 = r20 * r23;
    r85 = fmaf(r68, r22, r85);
    r52 = r20 * r32;
    r85 = fmaf(r65, r52, r85);
    r85 = fmaf(r7, r85, r9 * r24);
    r24 = r28 * r29;
    r24 = fmaf(r25, r34, r68 * r24);
    r24 = r24 + r84;
    r85 = fmaf(r8, r24, r85);
    r80 = r61 + r80;
    r61 = r28 * r23;
    r61 = r61 * r25;
    r24 = r18 * r20;
    r80 = fmaf(r68, r24, r80);
    r80 = r80 + r61;
    r64 = r18 * r64;
    r64 = r64 * r73;
    r70 = r64 + r70;
    r70 = fmaf(r7, r70, r8 * r80);
    r34 = fmaf(r65, r34, r42);
    r34 = r34 + r86;
    r70 = fmaf(r9, r34, r70);
    r34 = fmaf(r70, r36, r85 * r76);
    r86 = r6 * r54;
    r65 = r28 * r70;
    r65 = fmaf(r57, r65, r85 * r78);
    r42 = r85 * r55;
    r65 = fmaf(r38, r42, r65);
    r80 = r28 * r5;
    r61 = r1 + r61;
    r61 = r61 + r67;
    r67 = r29 * r20;
    r1 = r20 * r32;
    r1 = fmaf(r25, r1, r68 * r67);
    r1 = r1 + r84;
    r1 = fmaf(r9, r1, r7 * r61);
    r74 = r64 + r74;
    r1 = fmaf(r8, r74, r1);
    r80 = r80 * r1;
    r65 = fmaf(r56, r80, r65);
    r86 = r86 * r0;
    r86 = r86 * r65;
    r34 = fmaf(r46, r86, r34);
    r86 = r6 * r54;
    r86 = r86 * r5;
    r86 = r86 * r65;
    r65 = r4 * r5;
    r65 = r65 * r85;
    r65 = r65 * r56;
    r65 = fmaf(r37, r65, r46 * r86);
    r65 = fmaf(r1, r36, r65);
    r1 = r6 * r54;
    r86 = r33 * r55;
    r80 = r28 * r58;
    r80 = r80 * r5;
    r80 = fmaf(r56, r80, r38 * r86);
    r86 = r28 * r44;
    r80 = fmaf(r57, r86, r80);
    r80 = fmaf(r33, r78, r80);
    r1 = r1 * r0;
    r1 = r1 * r80;
    r1 = fmaf(r46, r1, r44 * r36);
    r1 = fmaf(r33, r76, r1);
    r86 = r4 * r33;
    r86 = r86 * r5;
    r86 = r86 * r56;
    r86 = fmaf(r58, r36, r37 * r86);
    r42 = r6 * r54;
    r42 = r42 * r5;
    r42 = r42 * r80;
    r86 = fmaf(r46, r42, r86);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r65,
                                          r1,
                                          r86);
    r42 = r6 * r54;
    r80 = r49 * r55;
    r74 = r28 * r45;
    r74 = r74 * r5;
    r74 = fmaf(r56, r74, r38 * r80);
    r80 = r28 * r41;
    r74 = fmaf(r57, r80, r74);
    r74 = fmaf(r49, r78, r74);
    r42 = r42 * r0;
    r42 = r42 * r74;
    r42 = fmaf(r46, r42, r49 * r76);
    r42 = fmaf(r41, r36, r42);
    r80 = r6 * r54;
    r80 = r80 * r5;
    r80 = r80 * r74;
    r80 = fmaf(r45, r36, r46 * r80);
    r74 = r4 * r49;
    r74 = r74 * r5;
    r74 = r74 * r56;
    r80 = fmaf(r37, r74, r80);
    r74 = fmaf(r40, r36, r43 * r76);
    r8 = r6 * r54;
    r64 = r28 * r39;
    r64 = r64 * r5;
    r9 = r43 * r55;
    r9 = fmaf(r38, r9, r56 * r64);
    r64 = r28 * r40;
    r9 = fmaf(r57, r64, r9);
    r9 = fmaf(r43, r78, r9);
    r8 = r8 * r0;
    r8 = r8 * r9;
    r74 = fmaf(r46, r8, r74);
    r8 = r6 * r54;
    r8 = r8 * r5;
    r8 = r8 * r9;
    r8 = fmaf(r46, r8, r39 * r36);
    r9 = r4 * r43;
    r9 = r9 * r5;
    r9 = r9 * r56;
    r8 = fmaf(r37, r9, r8);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r42,
                                          r80,
                                          r74,
                                          r8);
    r9 = r4 * r3;
    r2 = r4 * r2;
    r9 = fmaf(r81, r2, r77 * r9);
    r64 = r4 * r3;
    r64 = fmaf(r75, r2, r71 * r64);
    r61 = r4 * r3;
    r61 = fmaf(r34, r2, r65 * r61);
    r7 = r4 * r3;
    r7 = fmaf(r1, r2, r86 * r7);
    WriteSum4<float, float>((float*)inout_shared, r9, r64, r61, r7);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r7 = r4 * r3;
    r7 = fmaf(r42, r2, r80 * r7);
    r61 = r4 * r3;
    r61 = fmaf(r74, r2, r8 * r61);
    WriteSum2<float, float>((float*)inout_shared, r7, r61);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r61 = fmaf(r81, r81, r77 * r77);
    r7 = fmaf(r75, r75, r71 * r71);
    r64 = fmaf(r65, r65, r34 * r34);
    r9 = fmaf(r1, r1, r86 * r86);
    WriteSum4<float, float>((float*)inout_shared, r61, r7, r64, r9);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = fmaf(r42, r42, r80 * r80);
    r64 = fmaf(r74, r74, r8 * r8);
    WriteSum2<float, float>((float*)inout_shared, r9, r64);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r64 = fmaf(r77, r71, r81 * r75);
    r9 = fmaf(r77, r65, r81 * r34);
    r7 = fmaf(r81, r1, r77 * r86);
    r61 = fmaf(r81, r42, r77 * r80);
    WriteSum4<float, float>((float*)inout_shared, r64, r9, r7, r61);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = fmaf(r77, r8, r81 * r74);
    r81 = fmaf(r71, r65, r75 * r34);
    r61 = fmaf(r75, r1, r71 * r86);
    r7 = fmaf(r75, r42, r71 * r80);
    WriteSum4<float, float>((float*)inout_shared, r77, r81, r61, r7);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r75 = fmaf(r75, r74, r71 * r8);
    r71 = fmaf(r34, r1, r65 * r86);
    r7 = fmaf(r34, r42, r65 * r80);
    r34 = fmaf(r34, r74, r65 * r8);
    WriteSum4<float, float>((float*)inout_shared, r75, r71, r7, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r1, r42, r86 * r80);
    r86 = fmaf(r86, r8, r1 * r74);
    r8 = fmaf(r80, r8, r42 * r74);
    WriteSum3<float, float>((float*)inout_shared, r34, r86, r8);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = r4 * r3;
    WriteSum2<float, float>((float*)inout_shared, r2, r8);
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
  if (global_thread_idx < problem_size) {
    r21 = fmaf(r47, r76, r27 * r36);
    r8 = r6 * r54;
    r86 = r28 * r30;
    r86 = r86 * r5;
    r34 = r28 * r27;
    r34 = fmaf(r57, r34, r56 * r86);
    r86 = r47 * r55;
    r34 = fmaf(r38, r86, r34);
    r34 = fmaf(r47, r78, r34);
    r8 = r8 * r0;
    r8 = r8 * r34;
    r21 = fmaf(r46, r8, r21);
    r8 = r4 * r47;
    r8 = r8 * r5;
    r8 = r8 * r56;
    r8 = fmaf(r30, r36, r37 * r8);
    r86 = r6 * r54;
    r86 = r86 * r5;
    r86 = r86 * r34;
    r8 = fmaf(r46, r86, r8);
    r86 = fmaf(r51, r76, r31 * r36);
    r34 = r6 * r54;
    r80 = r51 * r55;
    r74 = r28 * r31;
    r74 = fmaf(r57, r74, r38 * r80);
    r80 = r28 * r19;
    r80 = r80 * r5;
    r74 = fmaf(r56, r80, r74);
    r74 = fmaf(r51, r78, r74);
    r34 = r34 * r0;
    r34 = r34 * r74;
    r86 = fmaf(r46, r34, r86);
    r34 = r4 * r51;
    r34 = r34 * r5;
    r34 = r34 * r56;
    r80 = r6 * r54;
    r80 = r80 * r5;
    r80 = r80 * r74;
    r80 = fmaf(r46, r80, r37 * r34);
    r80 = fmaf(r19, r36, r80);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r21,
                                          r8,
                                          r86,
                                          r80);
    r34 = r6 * r54;
    r74 = r28 * r48;
    r74 = r74 * r5;
    r42 = r26 * r55;
    r42 = fmaf(r38, r42, r56 * r74);
    r74 = r28 * r35;
    r42 = fmaf(r57, r74, r42);
    r42 = fmaf(r26, r78, r42);
    r34 = r34 * r0;
    r34 = r34 * r42;
    r34 = fmaf(r35, r36, r46 * r34);
    r34 = fmaf(r26, r76, r34);
    r76 = r4 * r26;
    r76 = r76 * r5;
    r76 = r76 * r56;
    r76 = fmaf(r37, r76, r48 * r36);
    r36 = r6 * r54;
    r36 = r36 * r5;
    r36 = r36 * r42;
    r76 = fmaf(r46, r36, r76);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r76);
    r36 = r4 * r3;
    r36 = fmaf(r21, r2, r8 * r36);
    r46 = r4 * r3;
    r46 = fmaf(r86, r2, r80 * r46);
    r42 = r4 * r3;
    r2 = fmaf(r34, r2, r76 * r42);
    WriteSum3<float, float>((float*)inout_shared, r36, r46, r2);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r21, r21, r8 * r8);
    r46 = fmaf(r86, r86, r80 * r80);
    r36 = fmaf(r76, r76, r34 * r34);
    WriteSum3<float, float>((float*)inout_shared, r2, r46, r36);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r36 = fmaf(r21, r86, r8 * r80);
    r21 = fmaf(r21, r34, r8 * r76);
    r76 = fmaf(r80, r76, r86 * r34);
    WriteSum3<float, float>((float*)inout_shared, r36, r21, r76);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndExtraResJac(
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
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraResJacKernel<<<n_blocks, 1024>>>(
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
      focal_and_extra,
      focal_and_extra_num_alloc,
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