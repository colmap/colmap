#include "kernel_simple_radial_split_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87;

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
    r21 = r46 * r37;
    r2 = fmaf(r0, r21, r2);
    r3 = fmaf(r3, r4, r1);
    r3 = fmaf(r5, r21, r3);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r2, r3);
    r1 = r28 * r23;
    r36 = -5.00000000000000000e-01;
    r52 = r11 * r36;
    r50 = 5.00000000000000000e-01;
    r42 = fmaf(r50, r25, r17 * r52);
    r42 = fmaf(r36, r22, r42);
    r42 = fmaf(r36, r24, r42);
    r1 = r1 * r42;
    r59 = r28 * r18;
    r60 = r11 * r14;
    r61 = r10 * r15;
    r61 = fmaf(r36, r61, r50 * r60);
    r60 = r13 * r16;
    r61 = fmaf(r50, r60, r61);
    r62 = r17 * r50;
    r61 = fmaf(r12, r62, r61);
    r59 = fmaf(r61, r59, r1);
    r60 = r28 * r29;
    r63 = r10 * r17;
    r64 = r13 * r14;
    r64 = fmaf(r36, r64, r36 * r63);
    r63 = r12 * r15;
    r64 = fmaf(r36, r63, r64);
    r65 = r11 * r16;
    r64 = fmaf(r50, r65, r64);
    r60 = r60 * r64;
    r65 = r10 * r14;
    r63 = r12 * r16;
    r63 = fmaf(r36, r63, r36 * r65);
    r63 = fmaf(r13, r62, r63);
    r63 = fmaf(r15, r52, r63);
    r65 = r63 * r34;
    r66 = r60 + r65;
    r67 = r59 + r66;
    r68 = r20 * r32;
    r69 = r23 * r64;
    r68 = fmaf(r20, r69, r61 * r68);
    r70 = r28 * r29;
    r71 = r28 * r18;
    r71 = r71 * r63;
    r70 = fmaf(r42, r70, r71);
    r68 = r68 + r70;
    r68 = fmaf(r7, r68, r8 * r67);
    r67 = r23 * r61;
    r72 = -4.00000000000000000e+00;
    r67 = r67 * r72;
    r73 = r29 * r72;
    r74 = r63 * r73;
    r75 = r67 + r74;
    r68 = fmaf(r9, r75, r68);
    r75 = r4 * r37;
    r75 = r75 * r57;
    r76 = r6 * r54;
    r77 = r28 * r5;
    r78 = r29 * r20;
    r79 = r20 * r32;
    r79 = r79 * r63;
    r78 = fmaf(r64, r78, r79);
    r78 = r78 + r59;
    r59 = r18 * r42;
    r80 = r72 * r59;
    r74 = r74 + r80;
    r74 = fmaf(r8, r74, r9 * r78);
    r78 = r28 * r29;
    r78 = r78 * r61;
    r81 = fmaf(r42, r34, r78);
    r82 = r28 * r23;
    r82 = r82 * r63;
    r83 = r28 * r18;
    r83 = fmaf(r64, r83, r82);
    r81 = r81 + r83;
    r74 = fmaf(r7, r81, r74);
    r77 = r77 * r74;
    r61 = fmaf(r61, r34, r28 * r69);
    r61 = r61 + r70;
    r78 = r82 + r78;
    r82 = r18 * r20;
    r78 = fmaf(r64, r82, r78);
    r81 = r20 * r32;
    r78 = fmaf(r42, r81, r78);
    r78 = fmaf(r8, r78, r9 * r61);
    r80 = r67 + r80;
    r78 = fmaf(r7, r80, r78);
    r80 = r28 * r78;
    r80 = fmaf(r57, r80, r56 * r77);
    r55 = r53 * r55;
    r55 = 1.0 / r55;
    r55 = r20 * r55;
    r53 = r68 * r55;
    r80 = fmaf(r38, r53, r80);
    r77 = r0 * r0;
    r77 = r77 * r55;
    r80 = fmaf(r68, r77, r80);
    r76 = r76 * r80;
    r76 = r76 * r46;
    r80 = fmaf(r0, r76, r68 * r75);
    r80 = fmaf(r78, r21, r80);
    r53 = r4 * r5;
    r53 = r53 * r68;
    r53 = r53 * r56;
    r76 = fmaf(r5, r76, r37 * r53);
    r76 = fmaf(r74, r21, r76);
    r74 = r6 * r54;
    r65 = r1 + r65;
    r1 = r28 * r18;
    r53 = r12 * r17;
    r67 = r10 * r15;
    r67 = fmaf(r50, r67, r36 * r53);
    r53 = r13 * r16;
    r67 = fmaf(r36, r53, r67);
    r67 = fmaf(r14, r52, r67);
    r1 = r1 * r67;
    r53 = r28 * r29;
    r61 = r13 * r14;
    r81 = r12 * r15;
    r81 = fmaf(r50, r81, r50 * r61);
    r81 = fmaf(r10, r62, r81);
    r81 = fmaf(r16, r52, r81);
    r53 = fmaf(r81, r53, r1);
    r65 = r65 + r53;
    r52 = r23 * r63;
    r52 = r52 * r72;
    r61 = r18 * r72;
    r61 = r61 * r81;
    r82 = r52 + r61;
    r82 = fmaf(r7, r82, r9 * r65);
    r65 = r20 * r32;
    r65 = fmaf(r20, r59, r81 * r65);
    r84 = r28 * r29;
    r84 = r84 * r63;
    r85 = r28 * r23;
    r85 = fmaf(r67, r85, r84);
    r65 = r65 + r85;
    r82 = fmaf(r8, r65, r82);
    r65 = r28 * r82;
    r86 = r20 * r23;
    r86 = fmaf(r42, r86, r79);
    r86 = r86 + r53;
    r53 = r28 * r23;
    r53 = r53 * r81;
    r87 = fmaf(r67, r34, r53);
    r87 = r87 + r70;
    r87 = fmaf(r8, r87, r7 * r86);
    r86 = r67 * r73;
    r52 = r52 + r86;
    r87 = fmaf(r9, r52, r87);
    r65 = fmaf(r87, r77, r57 * r65);
    r52 = r87 * r55;
    r65 = fmaf(r38, r52, r65);
    r70 = r28 * r5;
    r53 = r71 + r53;
    r71 = r29 * r20;
    r53 = fmaf(r42, r71, r53);
    r42 = r20 * r32;
    r53 = fmaf(r67, r42, r53);
    r81 = fmaf(r81, r34, r28 * r59);
    r81 = r81 + r85;
    r81 = fmaf(r7, r81, r9 * r53);
    r86 = r61 + r86;
    r81 = fmaf(r8, r86, r81);
    r70 = r70 * r81;
    r65 = fmaf(r56, r70, r65);
    r74 = r74 * r0;
    r74 = r74 * r65;
    r74 = fmaf(r82, r21, r46 * r74);
    r74 = fmaf(r87, r75, r74);
    r70 = r4 * r5;
    r70 = r70 * r87;
    r70 = r70 * r56;
    r70 = fmaf(r37, r70, r81 * r21);
    r81 = r6 * r54;
    r81 = r81 * r5;
    r81 = r81 * r65;
    r70 = fmaf(r46, r81, r70);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r80,
                                          r76,
                                          r74,
                                          r70);
    r25 = fmaf(r36, r25, r11 * r62);
    r25 = fmaf(r50, r22, r25);
    r25 = fmaf(r50, r24, r25);
    r73 = r25 * r73;
    r69 = r72 * r69;
    r24 = r73 + r69;
    r50 = r28 * r18;
    r50 = r50 * r25;
    r84 = r84 + r50;
    r22 = r20 * r23;
    r84 = fmaf(r67, r22, r84);
    r36 = r20 * r32;
    r84 = fmaf(r64, r36, r84);
    r84 = fmaf(r7, r84, r9 * r24);
    r24 = r28 * r29;
    r24 = fmaf(r25, r34, r67 * r24);
    r24 = r24 + r83;
    r84 = fmaf(r8, r24, r84);
    r79 = r60 + r79;
    r60 = r28 * r23;
    r60 = r60 * r25;
    r24 = r18 * r20;
    r79 = fmaf(r67, r24, r79);
    r79 = r79 + r60;
    r63 = r18 * r63;
    r63 = r63 * r72;
    r69 = r63 + r69;
    r69 = fmaf(r7, r69, r8 * r79);
    r34 = fmaf(r64, r34, r50);
    r34 = r34 + r85;
    r69 = fmaf(r9, r34, r69);
    r34 = fmaf(r69, r21, r84 * r75);
    r85 = r6 * r54;
    r64 = r28 * r69;
    r64 = fmaf(r57, r64, r84 * r77);
    r50 = r84 * r55;
    r64 = fmaf(r38, r50, r64);
    r79 = r28 * r5;
    r60 = r1 + r60;
    r60 = r60 + r66;
    r66 = r29 * r20;
    r1 = r20 * r32;
    r1 = fmaf(r25, r1, r67 * r66);
    r1 = r1 + r83;
    r1 = fmaf(r9, r1, r7 * r60);
    r73 = r63 + r73;
    r1 = fmaf(r8, r73, r1);
    r79 = r79 * r1;
    r64 = fmaf(r56, r79, r64);
    r85 = r85 * r0;
    r85 = r85 * r64;
    r34 = fmaf(r46, r85, r34);
    r85 = r6 * r54;
    r85 = r85 * r5;
    r85 = r85 * r64;
    r64 = r4 * r5;
    r64 = r64 * r84;
    r64 = r64 * r56;
    r64 = fmaf(r37, r64, r46 * r85);
    r64 = fmaf(r1, r21, r64);
    r1 = r6 * r54;
    r85 = r33 * r55;
    r79 = r28 * r58;
    r79 = r79 * r5;
    r79 = fmaf(r56, r79, r38 * r85);
    r85 = r28 * r44;
    r79 = fmaf(r57, r85, r79);
    r79 = fmaf(r33, r77, r79);
    r1 = r1 * r0;
    r1 = r1 * r79;
    r1 = fmaf(r46, r1, r44 * r21);
    r1 = fmaf(r33, r75, r1);
    r85 = r4 * r33;
    r85 = r85 * r5;
    r85 = r85 * r56;
    r85 = fmaf(r58, r21, r37 * r85);
    r50 = r6 * r54;
    r50 = r50 * r5;
    r50 = r50 * r79;
    r85 = fmaf(r46, r50, r85);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r64,
                                          r1,
                                          r85);
    r50 = r6 * r54;
    r79 = r49 * r55;
    r73 = r28 * r45;
    r73 = r73 * r5;
    r73 = fmaf(r56, r73, r38 * r79);
    r79 = r28 * r41;
    r73 = fmaf(r57, r79, r73);
    r73 = fmaf(r49, r77, r73);
    r50 = r50 * r0;
    r50 = r50 * r73;
    r50 = fmaf(r46, r50, r49 * r75);
    r50 = fmaf(r41, r21, r50);
    r79 = r6 * r54;
    r79 = r79 * r5;
    r79 = r79 * r73;
    r79 = fmaf(r45, r21, r46 * r79);
    r73 = r4 * r49;
    r73 = r73 * r5;
    r73 = r73 * r56;
    r79 = fmaf(r37, r73, r79);
    r73 = fmaf(r40, r21, r43 * r75);
    r8 = r6 * r54;
    r63 = r28 * r39;
    r63 = r63 * r5;
    r9 = r43 * r55;
    r9 = fmaf(r38, r9, r56 * r63);
    r63 = r28 * r40;
    r9 = fmaf(r57, r63, r9);
    r9 = fmaf(r43, r77, r9);
    r8 = r8 * r0;
    r8 = r8 * r9;
    r73 = fmaf(r46, r8, r73);
    r8 = r6 * r54;
    r8 = r8 * r5;
    r8 = r8 * r9;
    r8 = fmaf(r46, r8, r39 * r21);
    r9 = r4 * r43;
    r9 = r9 * r5;
    r9 = r9 * r56;
    r8 = fmaf(r37, r9, r8);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r50,
                                          r79,
                                          r73,
                                          r8);
    r9 = r4 * r2;
    r63 = r4 * r3;
    r63 = fmaf(r76, r63, r80 * r9);
    r9 = r4 * r2;
    r60 = r4 * r3;
    r60 = fmaf(r70, r60, r74 * r9);
    r9 = r4 * r2;
    r7 = r4 * r3;
    r7 = fmaf(r64, r7, r34 * r9);
    r9 = r4 * r3;
    r83 = r4 * r2;
    r83 = fmaf(r1, r83, r85 * r9);
    WriteSum4<float, float>((float*)inout_shared, r63, r60, r7, r83);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r4 * r2;
    r7 = r4 * r3;
    r7 = fmaf(r79, r7, r50 * r83);
    r83 = r4 * r3;
    r60 = r4 * r2;
    r60 = fmaf(r73, r60, r8 * r83);
    WriteSum2<float, float>((float*)inout_shared, r7, r60);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = fmaf(r80, r80, r76 * r76);
    r7 = fmaf(r74, r74, r70 * r70);
    r83 = fmaf(r64, r64, r34 * r34);
    r63 = fmaf(r1, r1, r85 * r85);
    WriteSum4<float, float>((float*)inout_shared, r60, r7, r83, r63);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = fmaf(r50, r50, r79 * r79);
    r83 = fmaf(r73, r73, r8 * r8);
    WriteSum2<float, float>((float*)inout_shared, r63, r83);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fmaf(r76, r70, r80 * r74);
    r63 = fmaf(r76, r64, r80 * r34);
    r7 = fmaf(r80, r1, r76 * r85);
    r60 = fmaf(r80, r50, r76 * r79);
    WriteSum4<float, float>((float*)inout_shared, r83, r63, r7, r60);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = fmaf(r76, r8, r80 * r73);
    r80 = fmaf(r70, r64, r74 * r34);
    r60 = fmaf(r74, r1, r70 * r85);
    r7 = fmaf(r74, r50, r70 * r79);
    WriteSum4<float, float>((float*)inout_shared, r76, r80, r60, r7);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r74 = fmaf(r74, r73, r70 * r8);
    r70 = fmaf(r34, r1, r64 * r85);
    r7 = fmaf(r34, r50, r64 * r79);
    r34 = fmaf(r34, r73, r64 * r8);
    WriteSum4<float, float>((float*)inout_shared, r74, r70, r7, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r1, r50, r85 * r79);
    r85 = fmaf(r85, r8, r1 * r73);
    r8 = fmaf(r79, r8, r50 * r73);
    WriteSum3<float, float>((float*)inout_shared, r34, r85, r8);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r8 = fmaf(r47, r75, r27 * r21);
    r85 = r6 * r54;
    r34 = r28 * r30;
    r34 = r34 * r5;
    r79 = r28 * r27;
    r79 = fmaf(r57, r79, r56 * r34);
    r34 = r47 * r55;
    r79 = fmaf(r38, r34, r79);
    r79 = fmaf(r47, r77, r79);
    r85 = r85 * r0;
    r85 = r85 * r79;
    r8 = fmaf(r46, r85, r8);
    r85 = r4 * r47;
    r85 = r85 * r5;
    r85 = r85 * r56;
    r85 = fmaf(r30, r21, r37 * r85);
    r34 = r6 * r54;
    r34 = r34 * r5;
    r34 = r34 * r79;
    r85 = fmaf(r46, r34, r85);
    r34 = fmaf(r51, r75, r31 * r21);
    r79 = r6 * r54;
    r73 = r51 * r55;
    r50 = r28 * r31;
    r50 = fmaf(r57, r50, r38 * r73);
    r73 = r28 * r19;
    r73 = r73 * r5;
    r50 = fmaf(r56, r73, r50);
    r50 = fmaf(r51, r77, r50);
    r79 = r79 * r0;
    r79 = r79 * r50;
    r34 = fmaf(r46, r79, r34);
    r79 = r4 * r51;
    r79 = r79 * r5;
    r79 = r79 * r56;
    r73 = r6 * r54;
    r73 = r73 * r5;
    r73 = r73 * r50;
    r73 = fmaf(r46, r73, r37 * r79);
    r73 = fmaf(r19, r21, r73);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r8,
                                          r85,
                                          r34,
                                          r73);
    r79 = r6 * r54;
    r50 = r28 * r48;
    r50 = r50 * r5;
    r1 = r26 * r55;
    r1 = fmaf(r38, r1, r56 * r50);
    r50 = r28 * r35;
    r1 = fmaf(r57, r50, r1);
    r1 = fmaf(r26, r77, r1);
    r79 = r79 * r0;
    r79 = r79 * r1;
    r79 = fmaf(r35, r21, r46 * r79);
    r79 = fmaf(r26, r75, r79);
    r75 = r4 * r26;
    r75 = r75 * r5;
    r75 = r75 * r56;
    r75 = fmaf(r37, r75, r48 * r21);
    r21 = r6 * r54;
    r21 = r21 * r5;
    r21 = r21 * r1;
    r75 = fmaf(r46, r21, r75);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r79,
                                          r75);
    r21 = r4 * r3;
    r46 = r4 * r2;
    r46 = fmaf(r8, r46, r85 * r21);
    r21 = r4 * r2;
    r1 = r4 * r3;
    r1 = fmaf(r73, r1, r34 * r21);
    r21 = r4 * r3;
    r37 = r4 * r2;
    r37 = fmaf(r79, r37, r75 * r21);
    WriteSum3<float, float>((float*)inout_shared, r46, r1, r37);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r37 = fmaf(r8, r8, r85 * r85);
    r1 = fmaf(r34, r34, r73 * r73);
    r46 = fmaf(r75, r75, r79 * r79);
    WriteSum3<float, float>((float*)inout_shared, r37, r1, r46);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r46 = fmaf(r8, r34, r85 * r73);
    r8 = fmaf(r8, r79, r85 * r75);
    r75 = fmaf(r73, r75, r34 * r79);
    WriteSum3<float, float>((float*)inout_shared, r46, r8, r75);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJac(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
  SimpleRadialSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                       1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      principal_point,
      principal_point_num_alloc,
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