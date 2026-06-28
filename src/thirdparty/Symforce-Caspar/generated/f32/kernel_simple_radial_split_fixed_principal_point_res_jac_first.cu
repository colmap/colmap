#include "kernel_simple_radial_split_fixed_principal_point_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SimpleRadialSplitFixedPrincipalPointResJacFirstKernel(
        float* pose,
        unsigned int pose_num_alloc,
        SharedIndex* pose_indices,
        float* sensor_from_rig,
        unsigned int sensor_from_rig_num_alloc,
        float* focal_and_extra,
        unsigned int focal_and_extra_num_alloc,
        SharedIndex* focal_and_extra_indices,
        float* point,
        unsigned int point_num_alloc,
        SharedIndex* point_indices,
        float* pixel,
        unsigned int pixel_num_alloc,
        float* principal_point,
        unsigned int principal_point_num_alloc,
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
        float* out_focal_and_extra_jac,
        unsigned int out_focal_and_extra_jac_num_alloc,
        float* const out_focal_and_extra_njtr,
        unsigned int out_focal_and_extra_njtr_num_alloc,
        float* const out_focal_and_extra_precond_diag,
        unsigned int out_focal_and_extra_precond_diag_num_alloc,
        float* const out_focal_and_extra_precond_tril,
        unsigned int out_focal_and_extra_precond_tril_num_alloc,
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

  __shared__ SharedIndex focal_and_extra_indices_loc[1024];
  focal_and_extra_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? focal_and_extra_indices[global_thread_idx]
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91;

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
  };
  LoadShared<2, float, float>(focal_and_extra,
                              0 * focal_and_extra_num_alloc,
                              focal_and_extra_indices_loc,
                              (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared,
                       focal_and_extra_indices_loc[threadIdx.x].target,
                       r46,
                       r47);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r48 = 9.99999999999999955e-07;
    r49 = r20 * r23;
    r49 = fmaf(r32, r49, r33);
    r6 = fmaf(r7, r49, r6);
    r33 = r15 * r17;
    r33 = fmaf(r20, r33, r39);
    r43 = r21 + r43;
    r39 = r14 * r14;
    r39 = r39 * r20;
    r43 = r43 + r39;
    r50 = r15 * r16;
    r50 = r50 * r28;
    r51 = r14 * r17;
    r51 = fmaf(r28, r51, r50);
    r52 = r28 * r18;
    r52 = r52 * r23;
    r53 = fmaf(r29, r34, r52);
    r54 = r29 * r29;
    r54 = r54 * r20;
    r26 = r54 + r26;
    r6 = fmaf(r36, r33, r6);
    r6 = fmaf(r38, r43, r6);
    r6 = fmaf(r37, r51, r6);
    r6 = fmaf(r8, r53, r6);
    r6 = fmaf(r9, r26, r6);
    r55 = copysign(1.0, r6);
    r55 = fmaf(r48, r55, r6);
    r48 = r55 * r55;
    r6 = 1.0 / r48;
    r56 = r0 * r6;
    r30 = fmaf(r18, r34, r30);
    r5 = fmaf(r7, r30, r5);
    r57 = r16 * r17;
    r57 = fmaf(r28, r57, r42);
    r45 = r21 + r45;
    r45 = r45 + r39;
    r39 = r14 * r17;
    r39 = fmaf(r20, r39, r50);
    r50 = r29 * r20;
    r50 = fmaf(r32, r50, r52);
    r19 = r21 + r19;
    r19 = r19 + r54;
    r5 = fmaf(r36, r57, r5);
    r5 = fmaf(r37, r45, r5);
    r5 = fmaf(r38, r39, r5);
    r5 = fmaf(r9, r50, r5);
    r5 = fmaf(r8, r19, r5);
    r38 = r5 * r5;
    r37 = fmaf(r6, r38, r0 * r56);
    r21 = fmaf(r47, r37, r21);
    r36 = r0 * r21;
    r54 = 1.0 / r55;
    r52 = r46 * r54;
    r2 = fmaf(r52, r36, r2);
    r3 = fmaf(r3, r4, r1);
    r1 = r5 * r21;
    r3 = fmaf(r52, r1, r3);
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
    r1 = r28 * r23;
    r36 = -5.00000000000000000e-01;
    r42 = r11 * r36;
    r58 = 5.00000000000000000e-01;
    r59 = fmaf(r58, r25, r17 * r42);
    r59 = fmaf(r36, r22, r59);
    r59 = fmaf(r36, r24, r59);
    r1 = r1 * r59;
    r60 = r28 * r18;
    r61 = r11 * r14;
    r62 = r10 * r15;
    r62 = fmaf(r36, r62, r58 * r61);
    r61 = r13 * r16;
    r62 = fmaf(r58, r61, r62);
    r63 = r17 * r58;
    r62 = fmaf(r12, r63, r62);
    r60 = fmaf(r62, r60, r1);
    r61 = r28 * r29;
    r64 = r10 * r17;
    r65 = r13 * r14;
    r65 = fmaf(r36, r65, r36 * r64);
    r64 = r12 * r15;
    r65 = fmaf(r36, r64, r65);
    r66 = r11 * r16;
    r65 = fmaf(r58, r66, r65);
    r61 = r61 * r65;
    r66 = r10 * r14;
    r64 = r12 * r16;
    r64 = fmaf(r36, r64, r36 * r66);
    r64 = fmaf(r13, r63, r64);
    r64 = fmaf(r15, r42, r64);
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
    r76 = r69 * r56;
    r77 = r4 * r21;
    r78 = r46 * r77;
    r79 = r28 * r5;
    r80 = r29 * r20;
    r81 = r20 * r32;
    r81 = r81 * r64;
    r80 = fmaf(r65, r80, r81);
    r80 = r80 + r60;
    r60 = r18 * r59;
    r82 = r73 * r60;
    r75 = r75 + r82;
    r75 = fmaf(r8, r75, r9 * r80);
    r80 = r28 * r29;
    r80 = r80 * r62;
    r83 = fmaf(r59, r34, r80);
    r84 = r28 * r23;
    r84 = r84 * r64;
    r85 = r28 * r18;
    r85 = fmaf(r65, r85, r84);
    r83 = r83 + r85;
    r75 = fmaf(r7, r83, r75);
    r79 = r79 * r75;
    r62 = fmaf(r62, r34, r28 * r70);
    r62 = r62 + r71;
    r80 = r84 + r80;
    r84 = r18 * r20;
    r80 = fmaf(r65, r84, r80);
    r83 = r20 * r32;
    r80 = fmaf(r59, r83, r80);
    r80 = fmaf(r8, r80, r9 * r62);
    r82 = r68 + r82;
    r80 = fmaf(r7, r82, r80);
    r82 = r28 * r80;
    r82 = fmaf(r56, r82, r6 * r79);
    r48 = r55 * r48;
    r48 = 1.0 / r48;
    r48 = r20 * r48;
    r55 = r69 * r48;
    r82 = fmaf(r38, r55, r82);
    r79 = r0 * r0;
    r79 = r79 * r48;
    r82 = fmaf(r69, r79, r82);
    r55 = r0 * r82;
    r47 = r47 * r52;
    r55 = fmaf(r47, r55, r78 * r76);
    r76 = r21 * r80;
    r55 = fmaf(r52, r76, r55);
    r76 = r5 * r6;
    r76 = r76 * r78;
    r68 = r5 * r82;
    r68 = fmaf(r47, r68, r69 * r76);
    r62 = r21 * r75;
    r68 = fmaf(r52, r62, r68);
    r66 = r1 + r66;
    r1 = r28 * r18;
    r62 = r12 * r17;
    r83 = r10 * r15;
    r83 = fmaf(r58, r83, r36 * r62);
    r62 = r13 * r16;
    r83 = fmaf(r36, r62, r83);
    r83 = fmaf(r14, r42, r83);
    r1 = r1 * r83;
    r62 = r28 * r29;
    r84 = r13 * r14;
    r86 = r12 * r15;
    r86 = fmaf(r58, r86, r58 * r84);
    r86 = fmaf(r10, r63, r86);
    r86 = fmaf(r16, r42, r86);
    r62 = fmaf(r86, r62, r1);
    r66 = r66 + r62;
    r42 = r23 * r64;
    r42 = r42 * r73;
    r84 = r18 * r73;
    r84 = r84 * r86;
    r87 = r42 + r84;
    r87 = fmaf(r7, r87, r9 * r66);
    r66 = r20 * r32;
    r66 = fmaf(r20, r60, r86 * r66);
    r88 = r28 * r29;
    r88 = r88 * r64;
    r89 = r28 * r23;
    r89 = fmaf(r83, r89, r88);
    r66 = r66 + r89;
    r87 = fmaf(r8, r66, r87);
    r66 = r28 * r87;
    r90 = r20 * r23;
    r90 = fmaf(r59, r90, r81);
    r90 = r90 + r62;
    r62 = r28 * r23;
    r62 = r62 * r86;
    r91 = fmaf(r83, r34, r62);
    r91 = r91 + r71;
    r91 = fmaf(r8, r91, r7 * r90);
    r90 = r83 * r74;
    r42 = r42 + r90;
    r91 = fmaf(r9, r42, r91);
    r66 = fmaf(r91, r79, r56 * r66);
    r42 = r91 * r48;
    r66 = fmaf(r38, r42, r66);
    r71 = r28 * r5;
    r62 = r72 + r62;
    r72 = r29 * r20;
    r62 = fmaf(r59, r72, r62);
    r59 = r20 * r32;
    r62 = fmaf(r83, r59, r62);
    r86 = fmaf(r86, r34, r28 * r60);
    r86 = r86 + r89;
    r86 = fmaf(r7, r86, r9 * r62);
    r90 = r84 + r90;
    r86 = fmaf(r8, r90, r86);
    r71 = r71 * r86;
    r66 = fmaf(r6, r71, r66);
    r71 = r0 * r66;
    r42 = r21 * r87;
    r42 = fmaf(r52, r42, r47 * r71);
    r71 = r91 * r56;
    r42 = fmaf(r78, r71, r42);
    r71 = r21 * r86;
    r71 = fmaf(r91, r76, r52 * r71);
    r90 = r5 * r66;
    r71 = fmaf(r47, r90, r71);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r55,
                                          r68,
                                          r42,
                                          r71);
    r25 = fmaf(r36, r25, r11 * r63);
    r25 = fmaf(r58, r22, r25);
    r25 = fmaf(r58, r24, r25);
    r74 = r25 * r74;
    r70 = r73 * r70;
    r24 = r74 + r70;
    r58 = r28 * r18;
    r58 = r58 * r25;
    r88 = r88 + r58;
    r22 = r20 * r23;
    r88 = fmaf(r83, r22, r88);
    r36 = r20 * r32;
    r88 = fmaf(r65, r36, r88);
    r88 = fmaf(r7, r88, r9 * r24);
    r24 = r28 * r29;
    r24 = fmaf(r25, r34, r83 * r24);
    r24 = r24 + r85;
    r88 = fmaf(r8, r24, r88);
    r24 = r88 * r56;
    r81 = r61 + r81;
    r61 = r28 * r23;
    r61 = r61 * r25;
    r36 = r18 * r20;
    r81 = fmaf(r83, r36, r81);
    r81 = r81 + r61;
    r64 = r18 * r64;
    r64 = r64 * r73;
    r70 = r64 + r70;
    r70 = fmaf(r7, r70, r8 * r81);
    r34 = fmaf(r65, r34, r58);
    r34 = r34 + r89;
    r70 = fmaf(r9, r34, r70);
    r34 = r21 * r70;
    r34 = fmaf(r52, r34, r78 * r24);
    r24 = r28 * r70;
    r24 = fmaf(r56, r24, r88 * r79);
    r89 = r88 * r48;
    r24 = fmaf(r38, r89, r24);
    r65 = r28 * r5;
    r61 = r1 + r61;
    r61 = r61 + r67;
    r67 = r29 * r20;
    r1 = r20 * r32;
    r1 = fmaf(r25, r1, r83 * r67);
    r1 = r1 + r85;
    r1 = fmaf(r9, r1, r7 * r61);
    r74 = r64 + r74;
    r1 = fmaf(r8, r74, r1);
    r65 = r65 * r1;
    r24 = fmaf(r6, r65, r24);
    r65 = r0 * r24;
    r34 = fmaf(r47, r65, r34);
    r65 = r5 * r24;
    r65 = fmaf(r88, r76, r47 * r65);
    r89 = r21 * r1;
    r65 = fmaf(r52, r89, r65);
    r89 = r44 * r21;
    r74 = r33 * r48;
    r8 = r28 * r57;
    r8 = r8 * r5;
    r8 = fmaf(r6, r8, r38 * r74);
    r74 = r28 * r44;
    r8 = fmaf(r56, r74, r8);
    r8 = fmaf(r33, r79, r8);
    r74 = r0 * r8;
    r74 = fmaf(r47, r74, r52 * r89);
    r89 = r33 * r56;
    r74 = fmaf(r78, r89, r74);
    r89 = r57 * r21;
    r89 = fmaf(r52, r89, r33 * r76);
    r64 = r5 * r8;
    r89 = fmaf(r47, r64, r89);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r34,
                                          r65,
                                          r74,
                                          r89);
    r64 = r51 * r56;
    r9 = r51 * r48;
    r61 = r28 * r45;
    r61 = r61 * r5;
    r61 = fmaf(r6, r61, r38 * r9);
    r9 = r28 * r41;
    r61 = fmaf(r56, r9, r61);
    r61 = fmaf(r51, r79, r61);
    r9 = r0 * r61;
    r9 = fmaf(r47, r9, r78 * r64);
    r64 = r41 * r21;
    r9 = fmaf(r52, r64, r9);
    r64 = r5 * r61;
    r7 = r45 * r21;
    r7 = fmaf(r52, r7, r47 * r64);
    r7 = fmaf(r51, r76, r7);
    r64 = r43 * r56;
    r85 = r40 * r21;
    r85 = fmaf(r52, r85, r78 * r64);
    r64 = r28 * r39;
    r64 = r64 * r5;
    r67 = r43 * r48;
    r67 = fmaf(r38, r67, r6 * r64);
    r64 = r28 * r40;
    r67 = fmaf(r56, r64, r67);
    r67 = fmaf(r43, r79, r67);
    r64 = r0 * r67;
    r85 = fmaf(r47, r64, r85);
    r64 = r39 * r21;
    r25 = r5 * r67;
    r25 = fmaf(r47, r25, r52 * r64);
    r25 = fmaf(r43, r76, r25);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r9,
                                          r7,
                                          r85,
                                          r25);
    r64 = r4 * r2;
    r83 = r4 * r3;
    r83 = fmaf(r68, r83, r55 * r64);
    r64 = r4 * r2;
    r58 = r4 * r3;
    r58 = fmaf(r71, r58, r42 * r64);
    r64 = r4 * r2;
    r81 = r4 * r3;
    r81 = fmaf(r65, r81, r34 * r64);
    r64 = r4 * r3;
    r73 = r4 * r2;
    r73 = fmaf(r74, r73, r89 * r64);
    WriteSum4<float, float>((float*)inout_shared, r83, r58, r81, r73);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = r4 * r2;
    r81 = r4 * r3;
    r81 = fmaf(r7, r81, r9 * r73);
    r73 = r4 * r3;
    r58 = r4 * r2;
    r58 = fmaf(r85, r58, r25 * r73);
    WriteSum2<float, float>((float*)inout_shared, r81, r58);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r58 = fmaf(r55, r55, r68 * r68);
    r81 = fmaf(r42, r42, r71 * r71);
    r73 = fmaf(r65, r65, r34 * r34);
    r83 = fmaf(r74, r74, r89 * r89);
    WriteSum4<float, float>((float*)inout_shared, r58, r81, r73, r83);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = fmaf(r9, r9, r7 * r7);
    r73 = fmaf(r85, r85, r25 * r25);
    WriteSum2<float, float>((float*)inout_shared, r83, r73);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r73 = fmaf(r68, r71, r55 * r42);
    r83 = fmaf(r68, r65, r55 * r34);
    r81 = fmaf(r55, r74, r68 * r89);
    r58 = fmaf(r55, r9, r68 * r7);
    WriteSum4<float, float>((float*)inout_shared, r73, r83, r81, r58);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r68 = fmaf(r68, r25, r55 * r85);
    r55 = fmaf(r71, r65, r42 * r34);
    r58 = fmaf(r42, r74, r71 * r89);
    r81 = fmaf(r42, r9, r71 * r7);
    WriteSum4<float, float>((float*)inout_shared, r68, r55, r58, r81);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r42 = fmaf(r42, r85, r71 * r25);
    r71 = fmaf(r34, r74, r65 * r89);
    r81 = fmaf(r34, r9, r65 * r7);
    r34 = fmaf(r34, r85, r65 * r25);
    WriteSum4<float, float>((float*)inout_shared, r42, r71, r81, r34);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = fmaf(r74, r9, r89 * r7);
    r89 = fmaf(r89, r25, r74 * r85);
    r25 = fmaf(r7, r25, r9 * r85);
    WriteSum3<float, float>((float*)inout_shared, r34, r89, r25);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r25 = r0 * r21;
    r25 = r25 * r54;
    r89 = r5 * r21;
    r89 = r89 * r54;
    r34 = r0 * r37;
    r34 = r34 * r52;
    r7 = r5 * r37;
    r7 = r7 * r52;
    WriteIdx4<1024, float, float, float4>(out_focal_and_extra_jac,
                                          0 * out_focal_and_extra_jac_num_alloc,
                                          global_thread_idx,
                                          r25,
                                          r89,
                                          r34,
                                          r7);
    r7 = r5 * r3;
    r7 = r7 * r54;
    r34 = r0 * r2;
    r34 = r34 * r54;
    r34 = fmaf(r77, r34, r77 * r7);
    r7 = r4 * r5;
    r7 = r7 * r37;
    r7 = r7 * r3;
    r77 = r4 * r0;
    r77 = r77 * r37;
    r77 = r77 * r2;
    r77 = fmaf(r52, r77, r52 * r7);
    WriteSum2<float, float>((float*)inout_shared, r34, r77);
  };
  FlushSumShared<2, float>(out_focal_and_extra_njtr,
                           0 * out_focal_and_extra_njtr_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r77 = r21 * r6;
    r77 = r77 * r38;
    r34 = r0 * r21;
    r34 = r34 * r21;
    r34 = fmaf(r56, r34, r21 * r77);
    r7 = r0 * r56;
    r54 = r46 * r46;
    r89 = r37 * r37;
    r54 = r54 * r89;
    r89 = r6 * r38;
    r89 = fmaf(r54, r89, r54 * r7);
    WriteSum2<float, float>((float*)inout_shared, r34, r89);
  };
  FlushSumShared<2, float>(out_focal_and_extra_precond_diag,
                           0 * out_focal_and_extra_precond_diag_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r89 = r46 * r0;
    r89 = r89 * r37;
    r89 = r89 * r21;
    r34 = r46 * r37;
    r34 = fmaf(r77, r34, r56 * r89);
    WriteSum1<float, float>((float*)inout_shared, r34);
  };
  FlushSumShared<1, float>(out_focal_and_extra_precond_tril,
                           0 * out_focal_and_extra_precond_tril_num_alloc,
                           focal_and_extra_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r27 * r21;
    r89 = r49 * r56;
    r89 = fmaf(r78, r89, r52 * r34);
    r34 = r28 * r30;
    r34 = r34 * r5;
    r77 = r28 * r27;
    r77 = fmaf(r56, r77, r6 * r34);
    r34 = r49 * r48;
    r77 = fmaf(r38, r34, r77);
    r77 = fmaf(r49, r79, r77);
    r77 = r77 * r47;
    r89 = fmaf(r0, r77, r89);
    r34 = r30 * r21;
    r34 = fmaf(r52, r34, r49 * r76);
    r34 = fmaf(r5, r77, r34);
    r77 = r31 * r21;
    r7 = r53 * r56;
    r7 = fmaf(r78, r7, r52 * r77);
    r77 = r53 * r48;
    r54 = r28 * r31;
    r54 = fmaf(r56, r54, r38 * r77);
    r77 = r28 * r19;
    r77 = r77 * r5;
    r54 = fmaf(r6, r77, r54);
    r54 = fmaf(r53, r79, r54);
    r77 = r0 * r54;
    r7 = fmaf(r47, r77, r7);
    r77 = r5 * r54;
    r77 = fmaf(r47, r77, r53 * r76);
    r25 = r19 * r21;
    r77 = fmaf(r52, r25, r77);
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r89,
                                          r34,
                                          r7,
                                          r77);
    r25 = r28 * r50;
    r25 = r25 * r5;
    r85 = r26 * r48;
    r85 = fmaf(r38, r85, r6 * r25);
    r25 = r28 * r35;
    r85 = fmaf(r56, r25, r85);
    r85 = fmaf(r26, r79, r85);
    r25 = r0 * r85;
    r79 = r35 * r21;
    r79 = fmaf(r52, r79, r47 * r25);
    r25 = r26 * r56;
    r79 = fmaf(r78, r25, r79);
    r25 = r50 * r21;
    r76 = fmaf(r26, r76, r52 * r25);
    r25 = r5 * r85;
    r76 = fmaf(r47, r25, r76);
    WriteIdx2<1024, float, float, float2>(out_point_jac,
                                          4 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r79,
                                          r76);
    r25 = r4 * r3;
    r47 = r4 * r2;
    r47 = fmaf(r89, r47, r34 * r25);
    r25 = r4 * r2;
    r52 = r4 * r3;
    r52 = fmaf(r77, r52, r7 * r25);
    r25 = r4 * r3;
    r78 = r4 * r2;
    r78 = fmaf(r79, r78, r76 * r25);
    WriteSum3<float, float>((float*)inout_shared, r47, r52, r78);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r78 = fmaf(r89, r89, r34 * r34);
    r52 = fmaf(r7, r7, r77 * r77);
    r47 = fmaf(r76, r76, r79 * r79);
    WriteSum3<float, float>((float*)inout_shared, r78, r52, r47);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r47 = fmaf(r89, r7, r34 * r77);
    r89 = fmaf(r89, r79, r34 * r76);
    r76 = fmaf(r77, r76, r7 * r79);
    WriteSum3<float, float>((float*)inout_shared, r47, r89, r76);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void SimpleRadialSplitFixedPrincipalPointResJacFirst(
    float* pose,
    unsigned int pose_num_alloc,
    SharedIndex* pose_indices,
    float* sensor_from_rig,
    unsigned int sensor_from_rig_num_alloc,
    float* focal_and_extra,
    unsigned int focal_and_extra_num_alloc,
    SharedIndex* focal_and_extra_indices,
    float* point,
    unsigned int point_num_alloc,
    SharedIndex* point_indices,
    float* pixel,
    unsigned int pixel_num_alloc,
    float* principal_point,
    unsigned int principal_point_num_alloc,
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
    float* out_focal_and_extra_jac,
    unsigned int out_focal_and_extra_jac_num_alloc,
    float* const out_focal_and_extra_njtr,
    unsigned int out_focal_and_extra_njtr_num_alloc,
    float* const out_focal_and_extra_precond_diag,
    unsigned int out_focal_and_extra_precond_diag_num_alloc,
    float* const out_focal_and_extra_precond_tril,
    unsigned int out_focal_and_extra_precond_tril_num_alloc,
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
  SimpleRadialSplitFixedPrincipalPointResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
      pose_num_alloc,
      pose_indices,
      sensor_from_rig,
      sensor_from_rig_num_alloc,
      focal_and_extra,
      focal_and_extra_num_alloc,
      focal_and_extra_indices,
      point,
      point_num_alloc,
      point_indices,
      pixel,
      pixel_num_alloc,
      principal_point,
      principal_point_num_alloc,
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
      out_focal_and_extra_jac,
      out_focal_and_extra_jac_num_alloc,
      out_focal_and_extra_njtr,
      out_focal_and_extra_njtr_num_alloc,
      out_focal_and_extra_precond_diag,
      out_focal_and_extra_precond_diag_num_alloc,
      out_focal_and_extra_precond_tril,
      out_focal_and_extra_precond_tril_num_alloc,
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