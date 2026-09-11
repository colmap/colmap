#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_opencv_split_fixed_focal_and_extra_fixed_principal_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    OpencvSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel(
        float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
        float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
        float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
        float *pixel, unsigned int pixel_num_alloc, float *focal_and_extra,
        unsigned int focal_and_extra_num_alloc, float *principal_point,
        unsigned int principal_point_num_alloc, float *out_res,
        unsigned int out_res_num_alloc, float *out_pose_jac,
        unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
        unsigned int out_pose_njtr_num_alloc,
        float *const out_pose_precond_diag,
        unsigned int out_pose_precond_diag_num_alloc,
        float *const out_pose_precond_tril,
        unsigned int out_pose_precond_tril_num_alloc, float *out_point_jac,
        unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
        unsigned int out_point_njtr_num_alloc,
        float *const out_point_precond_diag,
        unsigned int out_point_precond_diag_num_alloc,
        float *const out_point_precond_tril,
        unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {
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
      r76, r77, r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90,
      r91, r92, r93, r94, r95, r96, r97, r98, r99, r100;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(principal_point,
                                         0 * principal_point_num_alloc,
                                         global_thread_idx, r0, r1);
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
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r15, r16, r17,
                       r18);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r19, r20, r21, r22);
    r23 = fmaf(r18, r19, r15 * r22);
    r24 = r16 * r21;
    r25 = -1.00000000000000000e+00;
    r23 = fmaf(r25, r24, r23);
    r23 = fmaf(r17, r20, r23);
    r24 = r14 * r23;
    r26 = r18 * r20;
    r27 = fmaf(r16, r22, r26);
    r28 = r15 * r21;
    r29 = r17 * r19;
    r27 = r27 + r28;
    r27 = fmaf(r25, r29, r27);
    r24 = r24 * r27;
    r30 = fmaf(r16, r19, r17 * r22);
    r31 = r15 * r20;
    r30 = fmaf(r25, r31, r30);
    r30 = fmaf(r18, r21, r30);
    r31 = r14 * r30;
    r32 = fmaf(r16, r20, r15 * r19);
    r32 = fmaf(r17, r21, r32);
    r32 = fmaf(r25, r32, r18 * r22);
    r31 = fmaf(r32, r31, r24);
    r9 = fmaf(r11, r31, r9);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r33, r34, r35);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r36 = r19 * r20;
    r36 = r36 * r14;
    r37 = r21 * r22;
    r37 = fmaf(r14, r37, r36);
    r38 = r19 * r19;
    r39 = -2.00000000000000000e+00;
    r38 = r38 * r39;
    r40 = 1.00000000000000000e+00;
    r41 = r21 * r21;
    r41 = fmaf(r39, r41, r40);
    r42 = r38 + r41;
    r43 = r20 * r21;
    r43 = r43 * r14;
    r44 = r19 * r22;
    r44 = fmaf(r39, r44, r43);
    r45 = r14 * r30;
    r45 = r45 * r27;
    r46 = r39 * r32;
    r47 = fmaf(r23, r46, r45);
    r48 = r30 * r30;
    r48 = r48 * r39;
    r49 = r40 + r48;
    r50 = r23 * r23;
    r50 = r50 * r39;
    r49 = r49 + r50;
    r9 = fmaf(r33, r37, r9);
    r9 = fmaf(r34, r42, r9);
    r9 = fmaf(r35, r44, r9);
    r9 = fmaf(r13, r47, r9);
    r9 = fmaf(r12, r49, r9);
    r51 = r9 * r9;
    r52 = 9.99999999999999955e-07;
    r53 = r14 * r30;
    r53 = r53 * r23;
    r54 = fmaf(r27, r46, r53);
    r10 = fmaf(r11, r54, r10);
    r55 = r19 * r21;
    r55 = r55 * r14;
    r56 = r20 * r22;
    r56 = fmaf(r39, r56, r55);
    r57 = r20 * r20;
    r57 = r57 * r39;
    r58 = r40 + r57;
    r58 = r58 + r38;
    r38 = r19 * r22;
    r38 = fmaf(r14, r38, r43);
    r43 = r14 * r23;
    r43 = fmaf(r32, r43, r45);
    r45 = r39 * r27;
    r45 = r45 * r27;
    r59 = r40 + r45;
    r59 = r59 + r50;
    r10 = fmaf(r33, r56, r10);
    r10 = fmaf(r35, r58, r10);
    r10 = fmaf(r34, r38, r10);
    r10 = fmaf(r12, r43, r10);
    r10 = fmaf(r13, r59, r10);
    r50 = copysign(1.0, r10);
    r50 = fmaf(r52, r50, r10);
    r52 = r50 * r50;
    r10 = 1.0 / r52;
    r51 = r51 * r10;
    r45 = r40 + r45;
    r45 = r45 + r48;
    r8 = fmaf(r11, r45, r8);
    r24 = fmaf(r30, r46, r24);
    r48 = r14 * r27;
    r48 = fmaf(r32, r48, r53);
    r53 = r20 * r22;
    r53 = fmaf(r14, r53, r55);
    r55 = r21 * r22;
    r55 = fmaf(r39, r55, r36);
    r41 = r57 + r41;
    r8 = fmaf(r12, r24, r8);
    r8 = fmaf(r13, r48, r8);
    r8 = fmaf(r35, r53, r8);
    r8 = fmaf(r34, r55, r8);
    r8 = fmaf(r33, r41, r8);
    r33 = 3.00000000000000000e+00;
    r34 = r8 * r33;
    r35 = r8 * r10;
    r34 = fmaf(r35, r34, r51);
    r57 = 1.0 / r50;
    r34 = fmaf(r8, r57, r7 * r34);
    r36 = r8 * r35;
    r51 = r36 + r51;
    r5 = r5 * r51;
    r51 = fmaf(r51, r5, r4 * r51);
    r40 = r51 * r57;
    r60 = r14 * r35;
    r61 = r6 * r60;
    r34 = fmaf(r8, r40, r34);
    r34 = fmaf(r9, r61, r34);
    r34 = fmaf(r2, r34, r0);
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r0, r62);
    r34 = fmaf(r0, r25, r34);
    r0 = r9 * r9;
    r0 = r0 * r33;
    r0 = fmaf(r10, r0, r36);
    r0 = fmaf(r9, r57, r6 * r0);
    r36 = r7 * r9;
    r0 = fmaf(r60, r36, r0);
    r0 = fmaf(r9, r40, r0);
    r0 = fmaf(r3, r0, r1);
    r0 = fmaf(r62, r25, r0);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r34, r0);
    r62 = 6.00000000000000000e+00;
    r1 = r14 * r32;
    r36 = r16 * r19;
    r63 = 5.00000000000000000e-01;
    r64 = r15 * r20;
    r65 = -5.00000000000000000e-01;
    r64 = fmaf(r65, r64, r63 * r36);
    r36 = r18 * r21;
    r64 = fmaf(r63, r36, r64);
    r66 = r22 * r63;
    r64 = fmaf(r17, r66, r64);
    r36 = r15 * r22;
    r67 = r18 * r19;
    r67 = fmaf(r65, r67, r65 * r36);
    r36 = r17 * r20;
    r67 = fmaf(r65, r36, r67);
    r68 = r16 * r21;
    r67 = fmaf(r63, r68, r67);
    r68 = r27 * r67;
    r1 = fmaf(r14, r68, r64 * r1);
    r36 = r14 * r23;
    r69 = r16 * r65;
    r70 = fmaf(r63, r29, r22 * r69);
    r70 = fmaf(r65, r26, r70);
    r70 = fmaf(r65, r28, r70);
    r71 = r14 * r30;
    r72 = r15 * r19;
    r73 = r17 * r21;
    r73 = fmaf(r65, r73, r65 * r72);
    r73 = fmaf(r18, r66, r73);
    r73 = fmaf(r20, r69, r73);
    r71 = r71 * r73;
    r36 = fmaf(r70, r36, r71);
    r1 = r1 + r36;
    r72 = r14 * r27;
    r72 = r72 * r73;
    r74 = r14 * r23;
    r74 = r74 * r64;
    r75 = r72 + r74;
    r76 = r30 * r39;
    r75 = fmaf(r67, r76, r75);
    r75 = fmaf(r70, r46, r75);
    r75 = fmaf(r12, r75, r13 * r1);
    r1 = r27 * r64;
    r76 = -4.00000000000000000e+00;
    r1 = r1 * r76;
    r77 = r30 * r70;
    r78 = r76 * r77;
    r79 = r1 + r78;
    r75 = fmaf(r11, r79, r75);
    r79 = r62 * r75;
    r80 = r8 * r8;
    r81 = r14 * r27;
    r81 = r81 * r70;
    r82 = r14 * r30;
    r82 = fmaf(r64, r82, r81);
    r83 = r14 * r23;
    r83 = r83 * r67;
    r84 = r14 * r32;
    r84 = r84 * r73;
    r85 = r83 + r84;
    r86 = r82 + r85;
    r64 = fmaf(r64, r46, r39 * r68);
    r64 = r64 + r36;
    r64 = fmaf(r11, r64, r12 * r86);
    r86 = r23 * r73;
    r86 = r86 * r76;
    r1 = r86 + r1;
    r64 = fmaf(r13, r1, r64);
    r1 = -6.00000000000000000e+00;
    r52 = r50 * r52;
    r52 = 1.0 / r52;
    r80 = r80 * r64;
    r80 = r80 * r1;
    r80 = fmaf(r52, r80, r35 * r79);
    r79 = r14 * r9;
    r50 = r23 * r39;
    r87 = r73 * r46;
    r50 = fmaf(r67, r50, r87);
    r50 = r50 + r82;
    r78 = r86 + r78;
    r78 = fmaf(r12, r78, r13 * r50);
    r50 = r14 * r32;
    r50 = fmaf(r70, r50, r74);
    r74 = r14 * r30;
    r74 = fmaf(r67, r74, r72);
    r50 = r50 + r74;
    r78 = fmaf(r11, r50, r78);
    r79 = r79 * r78;
    r50 = r39 * r9;
    r72 = r9 * r52;
    r50 = r50 * r64;
    r50 = fmaf(r72, r50, r10 * r79);
    r80 = r80 + r50;
    r79 = r25 * r51;
    r79 = r79 * r64;
    r79 = fmaf(r35, r79, r7 * r80);
    r80 = r6 * r64;
    r86 = r8 * r76;
    r86 = r86 * r72;
    r79 = fmaf(r86, r80, r79);
    r82 = r6 * r14;
    r82 = r82 * r9;
    r82 = r82 * r75;
    r79 = fmaf(r10, r82, r79);
    r88 = r25 * r64;
    r79 = fmaf(r35, r88, r79);
    r89 = r39 * r8;
    r89 = r89 * r8;
    r89 = r89 * r64;
    r89 = fmaf(r52, r89, r75 * r60);
    r50 = r50 + r89;
    r5 = r14 * r5;
    r50 = fmaf(r50, r5, r4 * r50);
    r90 = r8 * r50;
    r79 = fmaf(r57, r90, r79);
    r79 = fmaf(r78, r61, r79);
    r79 = fmaf(r75, r40, r79);
    r79 = fmaf(r75, r57, r79);
    r90 = r2 * r79;
    r88 = r9 * r78;
    r88 = r88 * r62;
    r82 = r9 * r1;
    r82 = r82 * r72;
    r88 = fmaf(r64, r82, r10 * r88);
    r88 = r88 + r89;
    r88 = fmaf(r78, r57, r6 * r88);
    r89 = r25 * r9;
    r89 = r89 * r64;
    r88 = fmaf(r10, r89, r88);
    r80 = r7 * r78;
    r88 = fmaf(r60, r80, r88);
    r91 = r9 * r50;
    r88 = fmaf(r57, r91, r88);
    r92 = r7 * r86;
    r93 = r7 * r14;
    r93 = r93 * r9;
    r93 = r93 * r75;
    r88 = fmaf(r10, r93, r88);
    r94 = r25 * r9;
    r94 = r94 * r51;
    r94 = r94 * r64;
    r88 = fmaf(r10, r94, r88);
    r88 = fmaf(r78, r40, r88);
    r88 = fmaf(r64, r92, r88);
    r94 = r3 * r88;
    r84 = r81 + r84;
    r81 = r14 * r30;
    r93 = r17 * r22;
    r91 = r15 * r20;
    r91 = fmaf(r63, r91, r65 * r93);
    r93 = r18 * r21;
    r91 = fmaf(r65, r93, r91);
    r91 = fmaf(r19, r69, r91);
    r81 = r81 * r91;
    r93 = r14 * r23;
    r80 = r18 * r19;
    r89 = r17 * r20;
    r89 = fmaf(r63, r89, r63 * r80);
    r89 = fmaf(r15, r66, r89);
    r89 = fmaf(r21, r69, r89);
    r93 = fmaf(r89, r93, r81);
    r84 = r84 + r93;
    r69 = r27 * r73;
    r69 = r69 * r76;
    r80 = r30 * r76;
    r80 = r80 * r89;
    r95 = r69 + r80;
    r95 = fmaf(r11, r95, r13 * r84);
    r84 = fmaf(r89, r46, r39 * r77);
    r96 = r14 * r23;
    r96 = r96 * r73;
    r97 = r14 * r27;
    r97 = fmaf(r91, r97, r96);
    r84 = r84 + r97;
    r95 = fmaf(r12, r84, r95);
    r84 = r62 * r95;
    r98 = r8 * r8;
    r99 = r39 * r27;
    r99 = fmaf(r70, r99, r87);
    r99 = r99 + r93;
    r93 = r14 * r27;
    r93 = r93 * r89;
    r100 = r14 * r32;
    r100 = fmaf(r91, r100, r93);
    r100 = r100 + r36;
    r100 = fmaf(r12, r100, r11 * r99);
    r99 = r23 * r91;
    r36 = r76 * r99;
    r69 = r69 + r36;
    r100 = fmaf(r13, r69, r100);
    r98 = r98 * r1;
    r98 = r98 * r100;
    r98 = fmaf(r52, r98, r35 * r84);
    r84 = r39 * r9;
    r84 = r84 * r100;
    r69 = r14 * r9;
    r93 = r71 + r93;
    r71 = r23 * r39;
    r93 = fmaf(r70, r71, r93);
    r93 = fmaf(r91, r46, r93);
    r71 = r14 * r32;
    r77 = fmaf(r14, r77, r89 * r71);
    r77 = r77 + r97;
    r77 = fmaf(r11, r77, r13 * r93);
    r36 = r80 + r36;
    r77 = fmaf(r12, r36, r77);
    r69 = r69 * r77;
    r69 = fmaf(r10, r69, r72 * r84);
    r98 = r98 + r69;
    r98 = fmaf(r95, r40, r7 * r98);
    r84 = r25 * r100;
    r98 = fmaf(r35, r84, r98);
    r36 = r6 * r100;
    r98 = fmaf(r86, r36, r98);
    r80 = r6 * r14;
    r80 = r80 * r9;
    r80 = r80 * r95;
    r98 = fmaf(r10, r80, r98);
    r93 = r25 * r51;
    r93 = r93 * r100;
    r98 = fmaf(r35, r93, r98);
    r71 = r39 * r8;
    r71 = r71 * r8;
    r71 = r71 * r100;
    r71 = fmaf(r52, r71, r95 * r60);
    r69 = r69 + r71;
    r69 = fmaf(r69, r5, r4 * r69);
    r89 = r8 * r69;
    r98 = fmaf(r57, r89, r98);
    r98 = fmaf(r95, r57, r98);
    r98 = fmaf(r77, r61, r98);
    r89 = r2 * r98;
    r93 = r9 * r62;
    r93 = r93 * r77;
    r93 = fmaf(r10, r93, r100 * r82);
    r93 = r93 + r71;
    r71 = r25 * r9;
    r71 = r71 * r51;
    r71 = r71 * r100;
    r71 = fmaf(r10, r71, r6 * r93);
    r93 = r9 * r69;
    r71 = fmaf(r57, r93, r71);
    r80 = r25 * r9;
    r80 = r80 * r100;
    r71 = fmaf(r10, r80, r71);
    r36 = r7 * r14;
    r36 = r36 * r9;
    r36 = r36 * r95;
    r71 = fmaf(r10, r36, r71);
    r84 = r7 * r77;
    r71 = fmaf(r60, r84, r71);
    r71 = fmaf(r77, r57, r71);
    r71 = fmaf(r100, r92, r71);
    r71 = fmaf(r77, r40, r71);
    r84 = r3 * r71;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r90, r94,
        r89, r84);
    r84 = r8 * r8;
    r89 = r23 * r76;
    r29 = fmaf(r65, r29, r16 * r66);
    r29 = fmaf(r63, r26, r29);
    r29 = fmaf(r63, r28, r29);
    r89 = r89 * r29;
    r68 = r76 * r68;
    r28 = r89 + r68;
    r63 = r14 * r30;
    r63 = r63 * r29;
    r96 = r96 + r63;
    r26 = r39 * r27;
    r96 = fmaf(r91, r26, r96);
    r96 = fmaf(r67, r46, r96);
    r96 = fmaf(r11, r96, r13 * r28);
    r28 = r14 * r32;
    r28 = fmaf(r14, r99, r29 * r28);
    r28 = r28 + r74;
    r96 = fmaf(r12, r28, r96);
    r84 = r84 * r1;
    r84 = r84 * r96;
    r28 = r14 * r27;
    r28 = r28 * r29;
    r83 = r83 + r28;
    r26 = r30 * r39;
    r83 = fmaf(r91, r26, r83);
    r83 = r83 + r87;
    r73 = r30 * r73;
    r73 = r73 * r76;
    r68 = r73 + r68;
    r68 = fmaf(r11, r68, r12 * r83);
    r83 = r14 * r32;
    r83 = fmaf(r67, r83, r63);
    r83 = r83 + r97;
    r68 = fmaf(r13, r83, r68);
    r83 = r62 * r68;
    r83 = fmaf(r35, r83, r52 * r84);
    r84 = r39 * r9;
    r84 = r84 * r96;
    r97 = r14 * r9;
    r28 = r81 + r28;
    r28 = r28 + r85;
    r46 = fmaf(r29, r46, r39 * r99);
    r46 = r46 + r74;
    r46 = fmaf(r13, r46, r11 * r28);
    r89 = r73 + r89;
    r46 = fmaf(r12, r89, r46);
    r97 = r97 * r46;
    r97 = fmaf(r10, r97, r72 * r84);
    r83 = r83 + r97;
    r83 = fmaf(r68, r57, r7 * r83);
    r84 = r25 * r51;
    r84 = r84 * r96;
    r83 = fmaf(r35, r84, r83);
    r89 = r25 * r96;
    r83 = fmaf(r35, r89, r83);
    r12 = r6 * r14;
    r12 = r12 * r9;
    r12 = r12 * r68;
    r83 = fmaf(r10, r12, r83);
    r73 = r6 * r96;
    r83 = fmaf(r86, r73, r83);
    r13 = r39 * r8;
    r13 = r13 * r8;
    r13 = r13 * r96;
    r13 = fmaf(r68, r60, r52 * r13);
    r97 = r97 + r13;
    r97 = fmaf(r97, r5, r4 * r97);
    r28 = r8 * r97;
    r83 = fmaf(r57, r28, r83);
    r83 = fmaf(r46, r61, r83);
    r83 = fmaf(r68, r40, r83);
    r28 = r2 * r83;
    r73 = r9 * r62;
    r73 = r73 * r46;
    r73 = fmaf(r10, r73, r96 * r82);
    r73 = r73 + r13;
    r73 = fmaf(r46, r57, r6 * r73);
    r13 = r7 * r14;
    r13 = r13 * r9;
    r13 = r13 * r68;
    r73 = fmaf(r10, r13, r73);
    r12 = r9 * r97;
    r73 = fmaf(r57, r12, r73);
    r89 = r7 * r46;
    r73 = fmaf(r60, r89, r73);
    r84 = r25 * r9;
    r84 = r84 * r96;
    r73 = fmaf(r10, r84, r73);
    r11 = r25 * r9;
    r11 = r11 * r51;
    r11 = r11 * r96;
    r73 = fmaf(r10, r11, r73);
    r73 = fmaf(r96, r92, r73);
    r73 = fmaf(r46, r40, r73);
    r11 = r3 * r73;
    r84 = r56 * r8;
    r84 = r84 * r8;
    r84 = r84 * r1;
    r89 = r41 * r62;
    r89 = fmaf(r35, r89, r52 * r84);
    r84 = r39 * r56;
    r84 = r84 * r9;
    r12 = r14 * r37;
    r12 = r12 * r9;
    r12 = fmaf(r10, r12, r72 * r84);
    r89 = r89 + r12;
    r89 = fmaf(r41, r57, r7 * r89);
    r84 = r6 * r14;
    r84 = r84 * r41;
    r84 = r84 * r9;
    r89 = fmaf(r10, r84, r89);
    r13 = r6 * r56;
    r89 = fmaf(r86, r13, r89);
    r74 = r39 * r56;
    r74 = r74 * r8;
    r74 = r74 * r8;
    r74 = fmaf(r41, r60, r52 * r74);
    r12 = r12 + r74;
    r12 = fmaf(r12, r5, r4 * r12);
    r29 = r8 * r12;
    r89 = fmaf(r57, r29, r89);
    r99 = r25 * r56;
    r89 = fmaf(r35, r99, r89);
    r85 = r25 * r56;
    r85 = r85 * r51;
    r89 = fmaf(r35, r85, r89);
    r89 = fmaf(r41, r40, r89);
    r89 = fmaf(r37, r61, r89);
    r85 = r2 * r89;
    r99 = r37 * r9;
    r99 = r99 * r62;
    r99 = fmaf(r10, r99, r56 * r82);
    r99 = r99 + r74;
    r99 = fmaf(r37, r57, r6 * r99);
    r74 = r25 * r56;
    r74 = r74 * r9;
    r74 = r74 * r51;
    r99 = fmaf(r10, r74, r99);
    r29 = r7 * r14;
    r29 = r29 * r41;
    r29 = r29 * r9;
    r99 = fmaf(r10, r29, r99);
    r13 = r9 * r12;
    r99 = fmaf(r57, r13, r99);
    r84 = r25 * r56;
    r84 = r84 * r9;
    r99 = fmaf(r10, r84, r99);
    r81 = r7 * r37;
    r99 = fmaf(r60, r81, r99);
    r99 = fmaf(r56, r92, r99);
    r99 = fmaf(r37, r40, r99);
    r81 = r3 * r99;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r28, r11,
        r85, r81);
    r81 = r38 * r8;
    r81 = r81 * r8;
    r81 = r81 * r1;
    r85 = r55 * r62;
    r85 = fmaf(r35, r85, r52 * r81);
    r81 = r39 * r38;
    r81 = r81 * r9;
    r11 = r14 * r42;
    r11 = r11 * r9;
    r11 = fmaf(r10, r11, r72 * r81);
    r85 = r85 + r11;
    r81 = r25 * r38;
    r81 = r81 * r51;
    r81 = fmaf(r35, r81, r7 * r85);
    r85 = r6 * r14;
    r85 = r85 * r55;
    r85 = r85 * r9;
    r81 = fmaf(r10, r85, r81);
    r28 = r39 * r38;
    r28 = r28 * r8;
    r28 = r28 * r8;
    r28 = fmaf(r55, r60, r52 * r28);
    r11 = r11 + r28;
    r11 = fmaf(r11, r5, r4 * r11);
    r84 = r8 * r11;
    r81 = fmaf(r57, r84, r81);
    r13 = r6 * r38;
    r81 = fmaf(r86, r13, r81);
    r29 = r25 * r38;
    r81 = fmaf(r35, r29, r81);
    r81 = fmaf(r55, r57, r81);
    r81 = fmaf(r42, r61, r81);
    r81 = fmaf(r55, r40, r81);
    r29 = r2 * r81;
    r13 = r42 * r9;
    r13 = r13 * r62;
    r13 = fmaf(r10, r13, r38 * r82);
    r13 = r13 + r28;
    r13 = fmaf(r42, r57, r6 * r13);
    r28 = r7 * r14;
    r28 = r28 * r55;
    r28 = r28 * r9;
    r13 = fmaf(r10, r28, r13);
    r84 = r7 * r42;
    r13 = fmaf(r60, r84, r13);
    r85 = r25 * r38;
    r85 = r85 * r9;
    r13 = fmaf(r10, r85, r13);
    r74 = r9 * r11;
    r13 = fmaf(r57, r74, r13);
    r63 = r25 * r38;
    r63 = r63 * r9;
    r63 = r63 * r51;
    r13 = fmaf(r10, r63, r13);
    r13 = fmaf(r42, r40, r13);
    r13 = fmaf(r38, r92, r13);
    r63 = r3 * r13;
    r74 = r53 * r62;
    r85 = r58 * r8;
    r85 = r85 * r8;
    r85 = r85 * r1;
    r85 = fmaf(r52, r85, r35 * r74);
    r74 = r14 * r44;
    r74 = r74 * r9;
    r84 = r39 * r58;
    r84 = r84 * r9;
    r84 = fmaf(r72, r84, r10 * r74);
    r85 = r85 + r84;
    r74 = r6 * r58;
    r74 = fmaf(r86, r74, r7 * r85);
    r85 = r39 * r58;
    r85 = r85 * r8;
    r85 = r85 * r8;
    r85 = fmaf(r52, r85, r53 * r60);
    r84 = r84 + r85;
    r84 = fmaf(r84, r5, r4 * r84);
    r28 = r8 * r84;
    r74 = fmaf(r57, r28, r74);
    r67 = r25 * r58;
    r67 = r67 * r51;
    r74 = fmaf(r35, r67, r74);
    r76 = r25 * r58;
    r74 = fmaf(r35, r76, r74);
    r87 = r6 * r14;
    r87 = r87 * r53;
    r87 = r87 * r9;
    r74 = fmaf(r10, r87, r74);
    r74 = fmaf(r53, r57, r74);
    r74 = fmaf(r44, r61, r74);
    r74 = fmaf(r53, r40, r74);
    r87 = r2 * r74;
    r76 = r44 * r9;
    r76 = r76 * r62;
    r76 = fmaf(r58, r82, r10 * r76);
    r76 = r76 + r85;
    r76 = fmaf(r58, r92, r6 * r76);
    r85 = r25 * r58;
    r85 = r85 * r9;
    r85 = r85 * r51;
    r76 = fmaf(r10, r85, r76);
    r67 = r7 * r44;
    r76 = fmaf(r60, r67, r76);
    r28 = r25 * r58;
    r28 = r28 * r9;
    r76 = fmaf(r10, r28, r76);
    r26 = r9 * r84;
    r76 = fmaf(r57, r26, r76);
    r91 = r7 * r14;
    r91 = r91 * r53;
    r91 = r91 * r9;
    r76 = fmaf(r10, r91, r76);
    r76 = fmaf(r44, r57, r76);
    r76 = fmaf(r44, r40, r76);
    r91 = r3 * r76;
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 8 * out_pose_jac_num_alloc, global_thread_idx, r29, r63,
        r87, r91);
    r91 = r3 * r25;
    r91 = r91 * r0;
    r87 = r2 * r25;
    r87 = r87 * r34;
    r91 = fmaf(r79, r87, r88 * r91);
    r34 = r3 * r25;
    r34 = r34 * r0;
    r34 = fmaf(r98, r87, r71 * r34);
    r63 = r3 * r25;
    r63 = r63 * r0;
    r63 = fmaf(r83, r87, r73 * r63);
    r29 = r3 * r25;
    r29 = r29 * r0;
    r29 = fmaf(r89, r87, r99 * r29);
    WriteSum4<float, float>((float *)inout_shared, r91, r34, r63, r29);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r3 * r25;
    r29 = r29 * r0;
    r29 = fmaf(r81, r87, r13 * r29);
    r63 = r3 * r25;
    r63 = r63 * r0;
    r63 = fmaf(r74, r87, r76 * r63);
    WriteSum2<float, float>((float *)inout_shared, r29, r63);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r63 = r3 * r3;
    r29 = r88 * r63;
    r34 = r2 * r2;
    r91 = r79 * r34;
    r79 = fmaf(r79, r91, r88 * r29);
    r88 = r98 * r98;
    r26 = r71 * r71;
    r26 = fmaf(r63, r26, r34 * r88);
    r88 = r73 * r73;
    r28 = r83 * r83;
    r28 = fmaf(r34, r28, r63 * r88);
    r88 = r89 * r89;
    r67 = r99 * r99;
    r67 = fmaf(r63, r67, r34 * r88);
    WriteSum4<float, float>((float *)inout_shared, r79, r26, r28, r67);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r13 * r13;
    r28 = r81 * r81;
    r28 = fmaf(r34, r28, r63 * r67);
    r67 = r76 * r76;
    r26 = r74 * r74;
    r26 = fmaf(r34, r26, r63 * r67);
    WriteSum2<float, float>((float *)inout_shared, r28, r26);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r26 = fmaf(r98, r91, r71 * r29);
    r28 = fmaf(r73, r29, r83 * r91);
    r67 = fmaf(r99, r29, r89 * r91);
    r79 = fmaf(r81, r91, r13 * r29);
    WriteSum4<float, float>((float *)inout_shared, r26, r28, r67, r79);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r91 = fmaf(r74, r91, r76 * r29);
    r29 = r98 * r83;
    r79 = r71 * r73;
    r79 = fmaf(r63, r79, r34 * r29);
    r29 = r71 * r99;
    r67 = r98 * r89;
    r67 = fmaf(r34, r67, r63 * r29);
    r29 = r71 * r13;
    r28 = r98 * r81;
    r28 = fmaf(r34, r28, r63 * r29);
    WriteSum4<float, float>((float *)inout_shared, r91, r79, r67, r28);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r28 = r98 * r74;
    r67 = r71 * r76;
    r67 = fmaf(r63, r67, r34 * r28);
    r28 = r83 * r89;
    r79 = r73 * r99;
    r79 = fmaf(r63, r79, r34 * r28);
    r28 = r83 * r81;
    r91 = r73 * r13;
    r91 = fmaf(r63, r91, r34 * r28);
    r28 = r83 * r74;
    r29 = r73 * r76;
    r29 = fmaf(r63, r29, r34 * r28);
    WriteSum4<float, float>((float *)inout_shared, r67, r79, r91, r29);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r29 = r99 * r13;
    r91 = r89 * r81;
    r91 = fmaf(r34, r91, r63 * r29);
    r29 = r99 * r76;
    r79 = r89 * r74;
    r79 = fmaf(r34, r79, r63 * r29);
    r29 = r81 * r74;
    r67 = r13 * r76;
    r67 = fmaf(r63, r67, r34 * r29);
    WriteSum3<float, float>((float *)inout_shared, r91, r79, r67);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r45 * r62;
    r79 = r54 * r8;
    r79 = r79 * r8;
    r79 = r79 * r1;
    r79 = fmaf(r52, r79, r35 * r67);
    r67 = r14 * r31;
    r67 = r67 * r9;
    r91 = r39 * r54;
    r91 = r91 * r9;
    r91 = fmaf(r72, r91, r10 * r67);
    r79 = r79 + r91;
    r79 = fmaf(r45, r57, r7 * r79);
    r67 = r6 * r54;
    r79 = fmaf(r86, r67, r79);
    r29 = r6 * r14;
    r29 = r29 * r45;
    r29 = r29 * r9;
    r79 = fmaf(r10, r29, r79);
    r28 = r25 * r54;
    r79 = fmaf(r35, r28, r79);
    r26 = r39 * r54;
    r26 = r26 * r8;
    r26 = r26 * r8;
    r26 = fmaf(r52, r26, r45 * r60);
    r91 = r91 + r26;
    r91 = fmaf(r91, r5, r4 * r91);
    r88 = r8 * r91;
    r79 = fmaf(r57, r88, r79);
    r85 = r25 * r54;
    r85 = r85 * r51;
    r79 = fmaf(r35, r85, r79);
    r79 = fmaf(r31, r61, r79);
    r79 = fmaf(r45, r40, r79);
    r85 = r2 * r79;
    r88 = r31 * r9;
    r88 = r88 * r62;
    r88 = fmaf(r54, r82, r10 * r88);
    r88 = r88 + r26;
    r88 = fmaf(r54, r92, r6 * r88);
    r26 = r7 * r31;
    r88 = fmaf(r60, r26, r88);
    r28 = r7 * r14;
    r28 = r28 * r45;
    r28 = r28 * r9;
    r88 = fmaf(r10, r28, r88);
    r29 = r25 * r54;
    r29 = r29 * r9;
    r88 = fmaf(r10, r29, r88);
    r67 = r9 * r91;
    r88 = fmaf(r57, r67, r88);
    r65 = r25 * r54;
    r65 = r65 * r9;
    r65 = r65 * r51;
    r88 = fmaf(r10, r65, r88);
    r88 = fmaf(r31, r57, r88);
    r88 = fmaf(r31, r40, r88);
    r65 = r3 * r88;
    r67 = r24 * r62;
    r29 = r43 * r8;
    r29 = r29 * r8;
    r29 = r29 * r1;
    r29 = fmaf(r52, r29, r35 * r67);
    r67 = r39 * r43;
    r67 = r67 * r9;
    r28 = r14 * r49;
    r28 = r28 * r9;
    r28 = fmaf(r10, r28, r72 * r67);
    r29 = r29 + r28;
    r67 = r6 * r43;
    r67 = fmaf(r86, r67, r7 * r29);
    r29 = r39 * r43;
    r29 = r29 * r8;
    r29 = r29 * r8;
    r29 = fmaf(r52, r29, r24 * r60);
    r28 = r28 + r29;
    r28 = fmaf(r28, r5, r4 * r28);
    r26 = r8 * r28;
    r67 = fmaf(r57, r26, r67);
    r66 = r6 * r14;
    r66 = r66 * r24;
    r66 = r66 * r9;
    r67 = fmaf(r10, r66, r67);
    r94 = r25 * r43;
    r94 = r94 * r51;
    r67 = fmaf(r35, r94, r67);
    r90 = r25 * r43;
    r67 = fmaf(r35, r90, r67);
    r67 = fmaf(r24, r57, r67);
    r67 = fmaf(r24, r40, r67);
    r67 = fmaf(r49, r61, r67);
    r90 = r2 * r67;
    r94 = r49 * r9;
    r94 = r94 * r62;
    r94 = fmaf(r10, r94, r43 * r82);
    r94 = r94 + r29;
    r94 = fmaf(r49, r57, r6 * r94);
    r29 = r9 * r28;
    r94 = fmaf(r57, r29, r94);
    r66 = r25 * r43;
    r66 = r66 * r9;
    r94 = fmaf(r10, r66, r94);
    r26 = r25 * r43;
    r26 = r26 * r9;
    r26 = r26 * r51;
    r94 = fmaf(r10, r26, r94);
    r36 = r7 * r14;
    r36 = r36 * r24;
    r36 = r36 * r9;
    r94 = fmaf(r10, r36, r94);
    r80 = r7 * r49;
    r94 = fmaf(r60, r80, r94);
    r94 = fmaf(r43, r92, r94);
    r94 = fmaf(r49, r40, r94);
    r80 = r3 * r94;
    WriteIdx4<1024, float, float, float4>(
        out_point_jac, 0 * out_point_jac_num_alloc, global_thread_idx, r85, r65,
        r90, r80);
    r80 = r59 * r8;
    r80 = r80 * r8;
    r80 = r80 * r1;
    r1 = r48 * r62;
    r1 = fmaf(r35, r1, r52 * r80);
    r80 = r14 * r47;
    r80 = r80 * r9;
    r90 = r39 * r59;
    r90 = r90 * r9;
    r90 = fmaf(r72, r90, r10 * r80);
    r1 = r1 + r90;
    r61 = fmaf(r47, r61, r7 * r1);
    r1 = r25 * r59;
    r61 = fmaf(r35, r1, r61);
    r80 = r6 * r59;
    r61 = fmaf(r86, r80, r61);
    r86 = r39 * r59;
    r86 = r86 * r8;
    r86 = r86 * r8;
    r86 = fmaf(r48, r60, r52 * r86);
    r90 = r90 + r86;
    r5 = fmaf(r90, r5, r4 * r90);
    r90 = r8 * r5;
    r61 = fmaf(r57, r90, r61);
    r4 = r6 * r14;
    r4 = r4 * r48;
    r4 = r4 * r9;
    r61 = fmaf(r10, r4, r61);
    r52 = r25 * r59;
    r52 = r52 * r51;
    r61 = fmaf(r35, r52, r61);
    r61 = fmaf(r48, r57, r61);
    r61 = fmaf(r48, r40, r61);
    r2 = r2 * r61;
    r52 = r47 * r9;
    r52 = r52 * r62;
    r82 = fmaf(r59, r82, r10 * r52);
    r82 = r82 + r86;
    r86 = r7 * r47;
    r86 = fmaf(r60, r86, r6 * r82);
    r82 = r25 * r59;
    r82 = r82 * r9;
    r86 = fmaf(r10, r82, r86);
    r60 = r25 * r59;
    r60 = r60 * r9;
    r60 = r60 * r51;
    r86 = fmaf(r10, r60, r86);
    r52 = r9 * r5;
    r86 = fmaf(r57, r52, r86);
    r4 = r7 * r14;
    r4 = r4 * r48;
    r4 = r4 * r9;
    r86 = fmaf(r10, r4, r86);
    r86 = fmaf(r47, r57, r86);
    r86 = fmaf(r59, r92, r86);
    r86 = fmaf(r47, r40, r86);
    r40 = r3 * r86;
    WriteIdx2<1024, float, float, float2>(
        out_point_jac, 4 * out_point_jac_num_alloc, global_thread_idx, r2, r40);
    r40 = r3 * r25;
    r40 = r40 * r0;
    r40 = fmaf(r79, r87, r88 * r40);
    r2 = r3 * r25;
    r2 = r2 * r0;
    r2 = fmaf(r67, r87, r94 * r2);
    r4 = r3 * r25;
    r4 = r4 * r0;
    r87 = fmaf(r61, r87, r86 * r4);
    WriteSum3<float, float>((float *)inout_shared, r40, r2, r87);
  };
  FlushSumShared<3, float>(out_point_njtr, 0 * out_point_njtr_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r87 = r88 * r88;
    r2 = r79 * r79;
    r2 = fmaf(r34, r2, r63 * r87);
    r87 = r67 * r67;
    r40 = r94 * r94;
    r40 = fmaf(r63, r40, r34 * r87);
    r87 = r61 * r61;
    r4 = r86 * r86;
    r4 = fmaf(r63, r4, r34 * r87);
    WriteSum3<float, float>((float *)inout_shared, r2, r40, r4);
  };
  FlushSumShared<3, float>(out_point_precond_diag,
                           0 * out_point_precond_diag_num_alloc,
                           point_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r4 = r88 * r94;
    r40 = r79 * r67;
    r40 = fmaf(r34, r40, r63 * r4);
    r4 = r88 * r86;
    r2 = r79 * r61;
    r2 = fmaf(r34, r2, r63 * r4);
    r4 = r67 * r61;
    r87 = r94 * r86;
    r87 = fmaf(r63, r87, r34 * r4);
    WriteSum3<float, float>((float *)inout_shared, r40, r2, r87);
  };
  FlushSumShared<3, float>(out_point_precond_tril,
                           0 * out_point_precond_tril_num_alloc,
                           point_indices_loc, (float *)inout_shared);
}

void OpencvSplitFixedFocalAndExtraFixedPrincipalPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *point, unsigned int point_num_alloc, SharedIndex *point_indices,
    float *pixel, unsigned int pixel_num_alloc, float *focal_and_extra,
    unsigned int focal_and_extra_num_alloc, float *principal_point,
    unsigned int principal_point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_point_jac,
    unsigned int out_point_jac_num_alloc, float *const out_point_njtr,
    unsigned int out_point_njtr_num_alloc, float *const out_point_precond_diag,
    unsigned int out_point_precond_diag_num_alloc,
    float *const out_point_precond_tril,
    unsigned int out_point_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  OpencvSplitFixedFocalAndExtraFixedPrincipalPointResJacKernel<<<n_blocks,
                                                                 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, point, point_num_alloc, point_indices, pixel,
      pixel_num_alloc, focal_and_extra, focal_and_extra_num_alloc,
      principal_point, principal_point_num_alloc, out_res, out_res_num_alloc,
      out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_point_jac, out_point_jac_num_alloc,
      out_point_njtr, out_point_njtr_num_alloc, out_point_precond_diag,
      out_point_precond_diag_num_alloc, out_point_precond_tril,
      out_point_precond_tril_num_alloc, problem_size);
}

} // namespace caspar