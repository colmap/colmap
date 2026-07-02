#include "kernel_spherical_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    SphericalResJacKernel(float* pose,
                          unsigned int pose_num_alloc,
                          SharedIndex* pose_indices,
                          float* sensor_from_rig,
                          unsigned int sensor_from_rig_num_alloc,
                          float* wh,
                          unsigned int wh_num_alloc,
                          float* point,
                          unsigned int point_num_alloc,
                          SharedIndex* point_indices,
                          float* pixel,
                          unsigned int pixel_num_alloc,
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(
        wh, 0 * wh_num_alloc, global_thread_idx, r0, r1);
    r2 = 5.00000000000000000e-01;
    r3 = 1.59154943091895346e-01;
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx,
                                         r4,
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
  if (global_thread_idx < problem_size) {
    r10 = -2.00000000000000000e+00;
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
    r19 = r13 * r15;
    r20 = -1.00000000000000000e+00;
    r19 = fmaf(r20, r19, r12 * r18);
    r19 = fmaf(r14, r16, r19);
    r19 = fmaf(r11, r17, r19);
    r21 = r10 * r19;
    r21 = r21 * r19;
    r22 = 1.00000000000000000e+00;
    r23 = r13 * r18;
    r24 = fmaf(r12, r15, r23);
    r25 = r14 * r17;
    r26 = r11 * r16;
    r24 = r24 + r25;
    r24 = fmaf(r20, r26, r24);
    r27 = r10 * r24;
    r27 = fmaf(r24, r27, r22);
    r28 = r21 + r27;
    r4 = fmaf(r7, r28, r4);
    r29 = fmaf(r14, r15, r11 * r18);
    r30 = r12 * r17;
    r29 = fmaf(r20, r30, r29);
    r29 = fmaf(r13, r16, r29);
    r30 = 2.00000000000000000e+00;
    r31 = r30 * r19;
    r32 = r29 * r31;
    r33 = fmaf(r12, r16, r11 * r15);
    r33 = fmaf(r13, r17, r33);
    r33 = fmaf(r20, r33, r14 * r18);
    r34 = r10 * r33;
    r35 = fmaf(r24, r34, r32);
    r36 = r24 * r30;
    r36 = r36 * r29;
    r37 = fmaf(r33, r31, r36);
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r38,
                       r39,
                       r40);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r41 = r15 * r17;
    r41 = r41 * r30;
    r42 = r16 * r18;
    r42 = fmaf(r30, r42, r41);
    r43 = r17 * r18;
    r44 = r15 * r16;
    r44 = r44 * r30;
    r43 = fmaf(r10, r43, r44);
    r45 = r17 * r17;
    r45 = r45 * r10;
    r46 = r22 + r45;
    r47 = r16 * r16;
    r47 = r47 * r10;
    r46 = r46 + r47;
    r4 = fmaf(r8, r35, r4);
    r4 = fmaf(r9, r37, r4);
    r4 = fmaf(r40, r42, r4);
    r4 = fmaf(r39, r43, r4);
    r4 = fmaf(r38, r46, r4);
    r48 = 9.99999999999999955e-07;
    r36 = fmaf(r19, r34, r36);
    r6 = fmaf(r7, r36, r6);
    r49 = r16 * r18;
    r49 = fmaf(r10, r49, r41);
    r47 = r22 + r47;
    r41 = r15 * r15;
    r41 = r10 * r41;
    r47 = r47 + r41;
    r50 = r16 * r17;
    r50 = r50 * r30;
    r51 = r15 * r18;
    r51 = fmaf(r30, r51, r50);
    r52 = r30 * r29;
    r53 = r24 * r31;
    r52 = fmaf(r33, r52, r53);
    r21 = r22 + r21;
    r54 = r10 * r29;
    r54 = r54 * r29;
    r21 = r21 + r54;
    r6 = fmaf(r38, r49, r6);
    r6 = fmaf(r40, r47, r6);
    r6 = fmaf(r39, r51, r6);
    r6 = fmaf(r8, r52, r6);
    r6 = fmaf(r9, r21, r6);
    r55 = copysignf(r48, r6);
    r55 = r55 + r6;
    r56 = atan2f(r4, r55);
    r56 = fmaf(r3, r56, r2);
    ReadIdx2<1024, float, float, float2>(
        pixel, 0 * pixel_num_alloc, global_thread_idx, r57, r58);
    r57 = fmaf(r57, r20, r0 * r56);
    r56 = -3.18309886183790691e-01;
    r59 = r24 * r30;
    r59 = fmaf(r33, r59, r32);
    r5 = fmaf(r7, r59, r5);
    r32 = r17 * r18;
    r32 = fmaf(r30, r32, r44);
    r45 = r22 + r45;
    r45 = r45 + r41;
    r41 = r15 * r18;
    r41 = fmaf(r10, r41, r50);
    r53 = fmaf(r29, r34, r53);
    r27 = r54 + r27;
    r5 = fmaf(r38, r32, r5);
    r5 = fmaf(r39, r45, r5);
    r5 = fmaf(r40, r41, r5);
    r5 = fmaf(r9, r53, r5);
    r5 = fmaf(r8, r27, r5);
    r40 = r20 * r5;
    r39 = r4 * r4;
    r38 = r48 + r39;
    r38 = fmaf(r6, r6, r38);
    r54 = sqrtf(r38);
    r48 = copysignf(r48, r54);
    r54 = r48 + r54;
    r40 = atan2f(r40, r54);
    r40 = fmaf(r56, r40, r2);
    r58 = fmaf(r58, r20, r1 * r40);
    WriteIdx2<1024, float, float, float2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r57, r58);
    r40 = 1.0 / r55;
    r48 = r30 * r33;
    r50 = r12 * r15;
    r50 = fmaf(r2, r23, r2 * r50);
    r22 = -5.00000000000000000e-01;
    r50 = fmaf(r22, r26, r50);
    r50 = fmaf(r2, r25, r50);
    r44 = r11 * r18;
    r60 = r14 * r15;
    r60 = fmaf(r22, r60, r22 * r44);
    r44 = r13 * r16;
    r60 = fmaf(r22, r44, r60);
    r61 = r12 * r17;
    r60 = fmaf(r2, r61, r60);
    r48 = fmaf(r60, r31, r50 * r48);
    r61 = r30 * r29;
    r44 = r13 * r15;
    r62 = r14 * r16;
    r62 = fmaf(r22, r62, r2 * r44);
    r44 = r11 * r17;
    r62 = fmaf(r22, r44, r62);
    r63 = r12 * r22;
    r62 = fmaf(r18, r63, r62);
    r44 = r24 * r30;
    r64 = r14 * r18;
    r65 = r11 * r15;
    r65 = fmaf(r22, r65, r2 * r64);
    r64 = r13 * r17;
    r65 = fmaf(r22, r64, r65);
    r65 = fmaf(r16, r63, r65);
    r44 = r44 * r65;
    r61 = fmaf(r62, r61, r44);
    r48 = r48 + r61;
    r64 = r30 * r29;
    r64 = r64 * r50;
    r66 = r10 * r24;
    r66 = fmaf(r60, r66, r64);
    r67 = r65 * r31;
    r66 = r66 + r67;
    r66 = fmaf(r62, r34, r66);
    r66 = fmaf(r8, r66, r9 * r48);
    r48 = r19 * r50;
    r68 = -4.00000000000000000e+00;
    r48 = r48 * r68;
    r69 = r24 * r62;
    r70 = r68 * r69;
    r71 = r48 + r70;
    r66 = fmaf(r7, r71, r66);
    r71 = r24 * r30;
    r72 = r62 * r31;
    r71 = fmaf(r50, r71, r72);
    r73 = r30 * r29;
    r73 = r73 * r60;
    r74 = r30 * r33;
    r74 = r74 * r65;
    r75 = r73 + r74;
    r76 = r71 + r75;
    r77 = r10 * r19;
    r50 = fmaf(r50, r34, r60 * r77);
    r50 = r50 + r61;
    r50 = fmaf(r7, r50, r8 * r76);
    r76 = r29 * r68;
    r77 = r65 * r76;
    r48 = r48 + r77;
    r50 = fmaf(r9, r48, r50);
    r48 = r20 * r4;
    r55 = r55 * r55;
    r78 = 1.0 / r55;
    r48 = r48 * r78;
    r78 = fmaf(r50, r48, r66 * r40);
    r79 = r3 * r78;
    r39 = r39 + r55;
    r80 = 1.0 / r39;
    r81 = r0 * r55;
    r79 = r79 * r80;
    r79 = r79 * r81;
    r82 = r30 * r4;
    r83 = r30 * r6;
    r83 = fmaf(r50, r83, r66 * r82);
    r82 = r2 * r5;
    r50 = r54 * r54;
    r66 = 1.0 / r50;
    r38 = rsqrtf(r38);
    r82 = r82 * r66;
    r82 = r82 * r38;
    r38 = r10 * r29;
    r66 = r65 * r34;
    r38 = fmaf(r60, r38, r66);
    r38 = r38 + r71;
    r77 = r70 + r77;
    r77 = fmaf(r8, r77, r9 * r38);
    r38 = r30 * r33;
    r38 = fmaf(r62, r38, r64);
    r64 = r24 * r30;
    r64 = fmaf(r60, r64, r67);
    r38 = r38 + r64;
    r77 = fmaf(r7, r38, r77);
    r38 = r20 * r77;
    r54 = 1.0 / r54;
    r38 = fmaf(r54, r38, r83 * r82);
    r83 = r56 * r38;
    r5 = fmaf(r5, r5, r50);
    r67 = 1.0 / r5;
    r70 = r1 * r50;
    r83 = r83 * r67;
    r83 = r83 * r70;
    r72 = r74 + r72;
    r74 = r24 * r30;
    r23 = fmaf(r15, r63, r22 * r23);
    r23 = fmaf(r2, r26, r23);
    r23 = fmaf(r22, r25, r23);
    r74 = r74 * r23;
    r25 = r30 * r29;
    r26 = r11 * r18;
    r71 = r14 * r15;
    r71 = fmaf(r2, r71, r2 * r26);
    r26 = r13 * r16;
    r71 = fmaf(r2, r26, r71);
    r71 = fmaf(r17, r63, r71);
    r25 = fmaf(r71, r25, r74);
    r72 = r72 + r25;
    r63 = r19 * r65;
    r63 = r63 * r68;
    r26 = r24 * r68;
    r26 = r26 * r71;
    r84 = r63 + r26;
    r84 = fmaf(r7, r84, r9 * r72);
    r72 = fmaf(r71, r34, r10 * r69);
    r85 = r30 * r29;
    r85 = r85 * r65;
    r86 = fmaf(r23, r31, r85);
    r72 = r72 + r86;
    r84 = fmaf(r8, r72, r84);
    r72 = r10 * r19;
    r72 = fmaf(r62, r72, r66);
    r72 = r72 + r25;
    r25 = r30 * r33;
    r87 = r71 * r31;
    r25 = fmaf(r23, r25, r87);
    r25 = r25 + r61;
    r25 = fmaf(r8, r25, r7 * r72);
    r72 = r23 * r76;
    r63 = r63 + r72;
    r25 = fmaf(r9, r63, r25);
    r63 = fmaf(r25, r48, r84 * r40);
    r61 = r3 * r63;
    r61 = r61 * r80;
    r61 = r61 * r81;
    r88 = r30 * r4;
    r89 = r30 * r6;
    r89 = fmaf(r25, r89, r84 * r88);
    r88 = r10 * r29;
    r88 = fmaf(r62, r88, r44);
    r88 = r88 + r87;
    r88 = fmaf(r23, r34, r88);
    r87 = r30 * r33;
    r69 = fmaf(r30, r69, r71 * r87);
    r69 = r69 + r86;
    r69 = fmaf(r7, r69, r9 * r88);
    r72 = r26 + r72;
    r69 = fmaf(r8, r72, r69);
    r72 = r20 * r69;
    r72 = fmaf(r54, r72, r89 * r82);
    r89 = r56 * r72;
    r89 = r89 * r67;
    r89 = r89 * r70;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          0 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r79,
                                          r83,
                                          r61,
                                          r89);
    r89 = r19 * r60;
    r89 = r89 * r68;
    r61 = r12 * r18;
    r83 = r13 * r15;
    r83 = fmaf(r22, r83, r2 * r61);
    r61 = r14 * r16;
    r83 = fmaf(r2, r61, r83);
    r22 = r11 * r17;
    r83 = fmaf(r2, r22, r83);
    r76 = r83 * r76;
    r22 = r89 + r76;
    r61 = r24 * r30;
    r61 = r61 * r83;
    r85 = r85 + r61;
    r2 = r10 * r19;
    r85 = fmaf(r23, r2, r85);
    r85 = fmaf(r60, r34, r85);
    r85 = fmaf(r7, r85, r9 * r22);
    r22 = r30 * r29;
    r2 = r30 * r33;
    r2 = fmaf(r83, r2, r23 * r22);
    r2 = r2 + r64;
    r85 = fmaf(r8, r2, r85);
    r2 = r10 * r24;
    r2 = fmaf(r23, r2, r73);
    r31 = r83 * r31;
    r2 = r2 + r66;
    r2 = r2 + r31;
    r65 = r24 * r65;
    r65 = r65 * r68;
    r89 = r89 + r65;
    r89 = fmaf(r7, r89, r8 * r2);
    r2 = r30 * r33;
    r2 = fmaf(r60, r2, r61);
    r2 = r2 + r86;
    r89 = fmaf(r9, r2, r89);
    r2 = fmaf(r89, r40, r85 * r48);
    r86 = r3 * r2;
    r86 = r86 * r80;
    r86 = r86 * r81;
    r61 = r30 * r6;
    r60 = r30 * r4;
    r60 = fmaf(r89, r60, r85 * r61);
    r31 = r74 + r31;
    r31 = r31 + r75;
    r75 = r10 * r29;
    r34 = fmaf(r83, r34, r23 * r75);
    r34 = r34 + r64;
    r34 = fmaf(r9, r34, r7 * r31);
    r76 = r65 + r76;
    r34 = fmaf(r8, r76, r34);
    r76 = r20 * r34;
    r76 = fmaf(r54, r76, r60 * r82);
    r60 = r56 * r76;
    r60 = r60 * r67;
    r60 = r60 * r70;
    r8 = fmaf(r46, r40, r49 * r48);
    r65 = r3 * r8;
    r65 = r65 * r80;
    r65 = r65 * r81;
    r9 = r20 * r32;
    r31 = r30 * r46;
    r7 = r30 * r49;
    r7 = fmaf(r6, r7, r4 * r31);
    r7 = fmaf(r7, r82, r54 * r9);
    r9 = r56 * r7;
    r9 = r9 * r67;
    r9 = r9 * r70;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          4 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r86,
                                          r60,
                                          r65,
                                          r9);
    r9 = fmaf(r43, r40, r51 * r48);
    r65 = r3 * r9;
    r65 = r65 * r80;
    r65 = r65 * r81;
    r60 = r20 * r45;
    r86 = r30 * r43;
    r31 = r30 * r51;
    r31 = fmaf(r6, r31, r4 * r86);
    r31 = fmaf(r31, r82, r54 * r60);
    r60 = r56 * r31;
    r60 = r60 * r67;
    r60 = r60 * r70;
    r86 = fmaf(r42, r40, r47 * r48);
    r64 = r3 * r86;
    r64 = r64 * r80;
    r64 = r64 * r81;
    r83 = r20 * r41;
    r75 = r30 * r42;
    r23 = r30 * r47;
    r23 = fmaf(r6, r23, r4 * r75);
    r23 = fmaf(r23, r82, r54 * r83);
    r83 = r56 * r23;
    r83 = r83 * r67;
    r83 = r83 * r70;
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx,
                                          r65,
                                          r60,
                                          r64,
                                          r83);
    r83 = -1.59154943091895346e-01;
    r83 = r57 * r83;
    r83 = r83 * r80;
    r83 = r83 * r81;
    r57 = 3.18309886183790691e-01;
    r57 = r58 * r57;
    r57 = r57 * r67;
    r57 = r57 * r70;
    r58 = fmaf(r38, r57, r78 * r83);
    r64 = fmaf(r63, r83, r72 * r57);
    r60 = fmaf(r2, r83, r76 * r57);
    r65 = fmaf(r7, r57, r8 * r83);
    WriteSum4<float, float>((float*)inout_shared, r58, r64, r60, r65);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r65 = fmaf(r31, r57, r9 * r83);
    r60 = fmaf(r86, r83, r23 * r57);
    WriteSum2<float, float>((float*)inout_shared, r65, r60);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r60 = r78 * r78;
    r65 = 2.53302959105844473e-02;
    r65 = r0 * r65;
    r39 = r39 * r39;
    r39 = 1.0 / r39;
    r65 = r65 * r39;
    r65 = r65 * r81;
    r65 = r65 * r55;
    r55 = r38 * r38;
    r39 = 1.01321183642337789e-01;
    r39 = r1 * r39;
    r5 = r5 * r5;
    r5 = 1.0 / r5;
    r39 = r39 * r5;
    r39 = r39 * r70;
    r39 = r39 * r50;
    r55 = fmaf(r39, r55, r65 * r60);
    r60 = r72 * r72;
    r50 = r63 * r63;
    r50 = fmaf(r65, r50, r39 * r60);
    r60 = r76 * r39;
    r5 = r2 * r65;
    r2 = fmaf(r2, r5, r76 * r60);
    r76 = r7 * r7;
    r1 = r8 * r8;
    r1 = fmaf(r65, r1, r39 * r76);
    WriteSum4<float, float>((float*)inout_shared, r55, r50, r2, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r9 * r9;
    r2 = r31 * r31;
    r2 = fmaf(r39, r2, r65 * r1);
    r1 = r23 * r23;
    r50 = r86 * r86;
    r50 = fmaf(r65, r50, r39 * r1);
    WriteSum2<float, float>((float*)inout_shared, r2, r50);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r50 = r38 * r72;
    r2 = r78 * r63;
    r2 = fmaf(r65, r2, r39 * r50);
    r50 = fmaf(r38, r60, r78 * r5);
    r1 = r78 * r8;
    r55 = r38 * r7;
    r55 = fmaf(r39, r55, r65 * r1);
    r1 = r78 * r9;
    r76 = r38 * r31;
    r76 = fmaf(r39, r76, r65 * r1);
    WriteSum4<float, float>((float*)inout_shared, r2, r50, r55, r76);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r76 = r38 * r23;
    r55 = r78 * r86;
    r55 = fmaf(r65, r55, r39 * r76);
    r76 = fmaf(r63, r5, r72 * r60);
    r50 = r63 * r8;
    r2 = r72 * r7;
    r2 = fmaf(r39, r2, r65 * r50);
    r50 = r72 * r31;
    r1 = r63 * r9;
    r1 = fmaf(r65, r1, r39 * r50);
    WriteSum4<float, float>((float*)inout_shared, r55, r76, r2, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = r63 * r86;
    r2 = r72 * r23;
    r2 = fmaf(r39, r2, r65 * r1);
    r1 = fmaf(r8, r5, r7 * r60);
    r76 = fmaf(r9, r5, r31 * r60);
    r5 = fmaf(r86, r5, r23 * r60);
    WriteSum4<float, float>((float*)inout_shared, r2, r1, r76, r5);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = r7 * r31;
    r76 = r8 * r9;
    r76 = fmaf(r65, r76, r39 * r5);
    r5 = r7 * r23;
    r1 = r8 * r86;
    r1 = fmaf(r65, r1, r39 * r5);
    r5 = r9 * r86;
    r2 = r31 * r23;
    r2 = fmaf(r39, r2, r65 * r5);
    WriteSum3<float, float>((float*)inout_shared, r76, r1, r2);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r2 = fmaf(r28, r40, r36 * r48);
    r1 = r3 * r2;
    r1 = r1 * r80;
    r1 = r1 * r81;
    r76 = r20 * r59;
    r5 = r30 * r28;
    r60 = r30 * r36;
    r60 = fmaf(r6, r60, r4 * r5);
    r60 = fmaf(r60, r82, r54 * r76);
    r76 = r56 * r60;
    r76 = r76 * r67;
    r76 = r76 * r70;
    r5 = fmaf(r52, r48, r35 * r40);
    r55 = r3 * r5;
    r55 = r55 * r80;
    r55 = r55 * r81;
    r50 = r20 * r27;
    r0 = r30 * r52;
    r64 = r30 * r35;
    r64 = fmaf(r4, r64, r6 * r0);
    r64 = fmaf(r64, r82, r54 * r50);
    r50 = r56 * r64;
    r50 = r50 * r67;
    r50 = r50 * r70;
    WriteIdx4<1024, float, float, float4>(out_point_jac,
                                          0 * out_point_jac_num_alloc,
                                          global_thread_idx,
                                          r1,
                                          r76,
                                          r55,
                                          r50);
    r48 = fmaf(r21, r48, r37 * r40);
    r3 = r3 * r48;
    r3 = r3 * r80;
    r3 = r3 * r81;
    r81 = r20 * r53;
    r80 = r30 * r37;
    r40 = r30 * r21;
    r40 = fmaf(r6, r40, r4 * r80);
    r82 = fmaf(r40, r82, r54 * r81);
    r56 = r56 * r82;
    r56 = r56 * r67;
    r56 = r56 * r70;
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r3, r56);
    r56 = fmaf(r2, r83, r60 * r57);
    r3 = fmaf(r64, r57, r5 * r83);
    r57 = fmaf(r82, r57, r48 * r83);
    WriteSum3<float, float>((float*)inout_shared, r56, r3, r57);
  };
  FlushSumShared<3, float>(out_point_njtr,
                           0 * out_point_njtr_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r57 = r60 * r60;
    r3 = r2 * r2;
    r3 = fmaf(r65, r3, r39 * r57);
    r57 = r5 * r5;
    r56 = r64 * r64;
    r56 = fmaf(r39, r56, r65 * r57);
    r57 = r48 * r48;
    r83 = r82 * r82;
    r83 = fmaf(r39, r83, r65 * r57);
    WriteSum3<float, float>((float*)inout_shared, r3, r56, r83);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r83 = r60 * r64;
    r56 = r2 * r5;
    r56 = fmaf(r65, r56, r39 * r83);
    r83 = r60 * r82;
    r3 = r2 * r48;
    r3 = fmaf(r65, r3, r39 * r83);
    r83 = r64 * r82;
    r57 = r5 * r48;
    r57 = fmaf(r65, r57, r39 * r83);
    WriteSum3<float, float>((float*)inout_shared, r56, r3, r57);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc,
                           (float*)inout_shared);
}

void SphericalResJac(float* pose,
                     unsigned int pose_num_alloc,
                     SharedIndex* pose_indices,
                     float* sensor_from_rig,
                     unsigned int sensor_from_rig_num_alloc,
                     float* wh,
                     unsigned int wh_num_alloc,
                     float* point,
                     unsigned int point_num_alloc,
                     SharedIndex* point_indices,
                     float* pixel,
                     unsigned int pixel_num_alloc,
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
  SphericalResJacKernel<<<n_blocks, 1024>>>(pose,
                                            pose_num_alloc,
                                            pose_indices,
                                            sensor_from_rig,
                                            sensor_from_rig_num_alloc,
                                            wh,
                                            wh_num_alloc,
                                            point,
                                            point_num_alloc,
                                            point_indices,
                                            pixel,
                                            pixel_num_alloc,
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