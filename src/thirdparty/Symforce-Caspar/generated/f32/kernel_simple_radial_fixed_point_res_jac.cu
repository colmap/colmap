#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

#include "kernel_simple_radial_fixed_point_res_jac.h"
#include "memops.cuh"

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1) SimpleRadialFixedPointResJacKernel(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *pixel, unsigned int pixel_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ SharedIndex calib_indices_loc[1024];
  calib_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? calib_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60,
      r61, r62, r63, r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75,
      r76, r77, r78, r79, r80, r81, r82, r83;
  LoadShared<4, float, float>(calib, 0 * calib_num_alloc, calib_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       calib_indices_loc[threadIdx.x].target, r0, r1, r2, r3);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, float, float, float2>(pixel, 0 * pixel_num_alloc,
                                         global_thread_idx, r4, r5);
    r6 = -1.00000000000000000e+00;
    r4 = fmaf(r4, r6, r2);
    ReadIdx3<1024, float, float, float4>(sensor_from_rig,
                                         4 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r2, r7, r8);
    ReadIdx3<1024, float, float, float4>(point, 0 * point_num_alloc,
                                         global_thread_idx, r9, r10, r11);
  };
  LoadShared<4, float, float>(pose, 0 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r12, r13, r14,
                       r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    ReadIdx4<1024, float, float, float4>(sensor_from_rig,
                                         0 * sensor_from_rig_num_alloc,
                                         global_thread_idx, r16, r17, r18, r19);
    r20 = fmaf(r13, r16, r14 * r19);
    r21 = r12 * r17;
    r20 = fmaf(r6, r21, r20);
    r20 = fmaf(r15, r18, r20);
    r21 = r20 * r20;
    r22 = -2.00000000000000000e+00;
    r21 = r21 * r22;
    r23 = 1.00000000000000000e+00;
    r24 = r15 * r17;
    r25 = fmaf(r13, r19, r24);
    r26 = r12 * r18;
    r27 = r14 * r16;
    r25 = r25 + r26;
    r25 = fmaf(r6, r27, r25);
    r28 = r22 * r25;
    r28 = fmaf(r25, r28, r23);
    r29 = r21 + r28;
    r29 = fmaf(r9, r29, r2);
    r2 = 2.00000000000000000e+00;
    r30 = fmaf(r15, r16, r12 * r19);
    r31 = r13 * r18;
    r30 = fmaf(r6, r31, r30);
    r30 = fmaf(r14, r17, r30);
    r31 = r2 * r30;
    r31 = r31 * r25;
    r32 = r20 * r22;
    r33 = fmaf(r13, r17, r12 * r16);
    r33 = fmaf(r14, r18, r33);
    r33 = fmaf(r6, r33, r15 * r19);
    r32 = fmaf(r33, r32, r31);
    r34 = r2 * r20;
    r34 = r34 * r30;
    r35 = r2 * r33;
    r36 = fmaf(r25, r35, r34);
  };
  LoadShared<3, float, float>(pose, 4 * pose_num_alloc, pose_indices_loc,
                              (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float *)inout_shared,
                       pose_indices_loc[threadIdx.x].target, r37, r38, r39);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r40 = r16 * r18;
    r40 = r40 * r2;
    r41 = r17 * r19;
    r41 = fmaf(r2, r41, r40);
    r42 = r18 * r19;
    r43 = r16 * r17;
    r43 = r43 * r2;
    r42 = fmaf(r22, r42, r43);
    r44 = r17 * r17;
    r44 = r44 * r22;
    r45 = r23 + r44;
    r46 = r18 * r18;
    r46 = r46 * r22;
    r45 = r45 + r46;
    r29 = fmaf(r10, r32, r29);
    r29 = fmaf(r11, r36, r29);
    r29 = fmaf(r39, r41, r29);
    r29 = fmaf(r38, r42, r29);
    r29 = fmaf(r37, r45, r29);
    r36 = 9.99999999999999955e-07;
    r32 = r22 * r25;
    r32 = fmaf(r33, r32, r34);
    r32 = fmaf(r9, r32, r8);
    r8 = r17 * r19;
    r8 = fmaf(r22, r8, r40);
    r44 = r23 + r44;
    r40 = r16 * r16;
    r40 = r40 * r22;
    r44 = r44 + r40;
    r34 = r17 * r18;
    r34 = r34 * r2;
    r47 = r16 * r19;
    r47 = fmaf(r2, r47, r34);
    r48 = r2 * r20;
    r48 = r48 * r25;
    r49 = fmaf(r30, r35, r48);
    r50 = r30 * r30;
    r50 = r50 * r22;
    r28 = r50 + r28;
    r32 = fmaf(r37, r8, r32);
    r32 = fmaf(r39, r44, r32);
    r32 = fmaf(r38, r47, r32);
    r32 = fmaf(r10, r49, r32);
    r32 = fmaf(r11, r28, r32);
    r28 = copysign(1.0, r32);
    r28 = fmaf(r36, r28, r32);
    r36 = r28 * r28;
    r32 = 1.0 / r36;
    r49 = r29 * r32;
    r31 = fmaf(r20, r35, r31);
    r31 = fmaf(r9, r31, r7);
    r7 = r18 * r19;
    r7 = fmaf(r2, r7, r43);
    r46 = r23 + r46;
    r46 = r46 + r40;
    r40 = r16 * r19;
    r40 = fmaf(r22, r40, r34);
    r34 = r30 * r22;
    r34 = fmaf(r33, r34, r48);
    r21 = r23 + r21;
    r21 = r21 + r50;
    r31 = fmaf(r37, r7, r31);
    r31 = fmaf(r38, r46, r31);
    r31 = fmaf(r39, r40, r31);
    r31 = fmaf(r11, r34, r31);
    r31 = fmaf(r10, r21, r31);
    r21 = r31 * r31;
    r34 = fmaf(r32, r21, r29 * r49);
    r39 = fmaf(r1, r34, r23);
    r38 = r29 * r39;
    r37 = 1.0 / r28;
    r50 = r0 * r37;
    r4 = fmaf(r50, r38, r4);
    r5 = fmaf(r5, r6, r3);
    r3 = r31 * r39;
    r5 = fmaf(r50, r3, r5);
    WriteIdx2<1024, float, float, float2>(out_res, 0 * out_res_num_alloc,
                                          global_thread_idx, r4, r5);
    r3 = r2 * r31;
    r38 = r30 * r22;
    r48 = r12 * r19;
    r43 = -5.00000000000000000e-01;
    r51 = r15 * r16;
    r51 = fmaf(r43, r51, r43 * r48);
    r48 = r14 * r17;
    r51 = fmaf(r43, r48, r51);
    r52 = r13 * r18;
    r53 = 5.00000000000000000e-01;
    r51 = fmaf(r53, r52, r51);
    r52 = r22 * r33;
    r48 = r12 * r16;
    r54 = r14 * r18;
    r54 = fmaf(r43, r54, r43 * r48);
    r48 = r19 * r53;
    r55 = r13 * r43;
    r54 = fmaf(r15, r48, r54);
    r54 = fmaf(r17, r55, r54);
    r52 = r52 * r54;
    r38 = fmaf(r51, r38, r52);
    r56 = r2 * r25;
    r57 = fmaf(r53, r27, r19 * r55);
    r57 = fmaf(r43, r24, r57);
    r57 = fmaf(r43, r26, r57);
    r56 = r56 * r57;
    r58 = r2 * r20;
    r59 = r13 * r16;
    r60 = r12 * r17;
    r60 = fmaf(r43, r60, r53 * r59);
    r59 = r15 * r18;
    r60 = fmaf(r53, r59, r60);
    r60 = fmaf(r14, r48, r60);
    r58 = fmaf(r60, r58, r56);
    r38 = r38 + r58;
    r59 = -4.00000000000000000e+00;
    r61 = r30 * r59;
    r62 = r54 * r61;
    r63 = r20 * r57;
    r64 = r59 * r63;
    r65 = r62 + r64;
    r65 = fmaf(r10, r65, r11 * r38);
    r38 = r2 * r30;
    r38 = r38 * r60;
    r66 = fmaf(r57, r35, r38);
    r67 = r2 * r25;
    r67 = r67 * r54;
    r68 = r2 * r20;
    r68 = fmaf(r51, r68, r67);
    r66 = r66 + r68;
    r65 = fmaf(r9, r66, r65);
    r3 = r3 * r65;
    r66 = r25 * r51;
    r69 = fmaf(r60, r35, r2 * r66);
    r70 = r2 * r30;
    r71 = r2 * r20;
    r71 = r71 * r54;
    r70 = fmaf(r57, r70, r71);
    r69 = r69 + r70;
    r38 = r67 + r38;
    r67 = r20 * r22;
    r38 = fmaf(r51, r67, r38);
    r72 = r22 * r33;
    r38 = fmaf(r57, r72, r38);
    r38 = fmaf(r10, r38, r11 * r69);
    r69 = r25 * r60;
    r69 = r69 * r59;
    r64 = r69 + r64;
    r38 = fmaf(r9, r64, r38);
    r64 = r2 * r38;
    r64 = fmaf(r49, r64, r32 * r3);
    r3 = r2 * r30;
    r3 = r3 * r51;
    r72 = r54 * r35;
    r67 = r3 + r72;
    r58 = r58 + r67;
    r73 = r22 * r33;
    r73 = fmaf(r22, r66, r60 * r73);
    r73 = r73 + r70;
    r73 = fmaf(r9, r73, r10 * r58);
    r62 = r69 + r62;
    r73 = fmaf(r11, r62, r73);
    r36 = r28 * r36;
    r36 = 1.0 / r36;
    r36 = r22 * r36;
    r28 = r73 * r36;
    r64 = fmaf(r21, r28, r64);
    r62 = r29 * r29;
    r62 = r62 * r36;
    r64 = fmaf(r73, r62, r64);
    r28 = r29 * r64;
    r1 = r1 * r50;
    r69 = r73 * r49;
    r58 = r6 * r39;
    r60 = r0 * r58;
    r69 = fmaf(r60, r69, r1 * r28);
    r28 = r39 * r38;
    r69 = fmaf(r50, r28, r69);
    r28 = r39 * r65;
    r74 = r31 * r32;
    r74 = r74 * r60;
    r28 = fmaf(r73, r74, r50 * r28);
    r75 = r31 * r64;
    r28 = fmaf(r1, r75, r28);
    r72 = r56 + r72;
    r56 = r2 * r20;
    r75 = r14 * r19;
    r76 = r12 * r17;
    r76 = fmaf(r53, r76, r43 * r75);
    r75 = r15 * r18;
    r76 = fmaf(r43, r75, r76);
    r76 = fmaf(r16, r55, r76);
    r56 = r56 * r76;
    r75 = r2 * r30;
    r77 = r15 * r16;
    r78 = r14 * r17;
    r78 = fmaf(r53, r78, r53 * r77);
    r78 = fmaf(r12, r48, r78);
    r78 = fmaf(r18, r55, r78);
    r75 = fmaf(r78, r75, r56);
    r72 = r72 + r75;
    r55 = r25 * r54;
    r55 = r55 * r59;
    r77 = r20 * r59;
    r77 = r77 * r78;
    r79 = r55 + r77;
    r79 = fmaf(r9, r79, r11 * r72);
    r72 = r22 * r33;
    r72 = fmaf(r22, r63, r78 * r72);
    r80 = r2 * r30;
    r80 = r80 * r54;
    r81 = r2 * r25;
    r81 = fmaf(r76, r81, r80);
    r72 = r72 + r81;
    r79 = fmaf(r10, r72, r79);
    r72 = r2 * r79;
    r82 = r22 * r25;
    r82 = fmaf(r57, r82, r52);
    r82 = r82 + r75;
    r75 = r2 * r25;
    r75 = r75 * r78;
    r83 = fmaf(r76, r35, r75);
    r83 = r83 + r70;
    r83 = fmaf(r10, r83, r9 * r82);
    r82 = r76 * r61;
    r55 = r55 + r82;
    r83 = fmaf(r11, r55, r83);
    r72 = fmaf(r83, r62, r49 * r72);
    r55 = r83 * r36;
    r72 = fmaf(r21, r55, r72);
    r70 = r2 * r31;
    r75 = r71 + r75;
    r71 = r30 * r22;
    r75 = fmaf(r57, r71, r75);
    r57 = r22 * r33;
    r75 = fmaf(r76, r57, r75);
    r78 = fmaf(r78, r35, r2 * r63);
    r78 = r78 + r81;
    r78 = fmaf(r9, r78, r11 * r75);
    r82 = r77 + r82;
    r78 = fmaf(r10, r82, r78);
    r70 = r70 * r78;
    r72 = fmaf(r32, r70, r72);
    r70 = r29 * r72;
    r55 = r83 * r49;
    r55 = fmaf(r60, r55, r1 * r70);
    r70 = r39 * r79;
    r55 = fmaf(r50, r70, r55);
    r70 = r39 * r78;
    r82 = r31 * r72;
    r82 = fmaf(r1, r82, r50 * r70);
    r82 = fmaf(r83, r74, r82);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 0 * out_pose_jac_num_alloc, global_thread_idx, r69, r28,
        r55, r82);
    r27 = fmaf(r43, r27, r13 * r48);
    r27 = fmaf(r53, r24, r27);
    r27 = fmaf(r53, r26, r27);
    r61 = r27 * r61;
    r66 = r59 * r66;
    r26 = r61 + r66;
    r53 = r2 * r20;
    r53 = r53 * r27;
    r80 = r80 + r53;
    r24 = r22 * r25;
    r80 = fmaf(r76, r24, r80);
    r43 = r22 * r33;
    r80 = fmaf(r51, r43, r80);
    r80 = fmaf(r9, r80, r11 * r26);
    r26 = r2 * r30;
    r26 = fmaf(r27, r35, r76 * r26);
    r26 = r26 + r68;
    r80 = fmaf(r10, r26, r80);
    r26 = r80 * r49;
    r3 = r52 + r3;
    r52 = r2 * r25;
    r52 = r52 * r27;
    r43 = r20 * r22;
    r3 = fmaf(r76, r43, r3);
    r3 = r3 + r52;
    r54 = r20 * r54;
    r54 = r54 * r59;
    r66 = r54 + r66;
    r66 = fmaf(r9, r66, r10 * r3);
    r35 = fmaf(r51, r35, r53);
    r35 = r35 + r81;
    r66 = fmaf(r11, r35, r66);
    r35 = r2 * r66;
    r35 = fmaf(r49, r35, r80 * r62);
    r81 = r80 * r36;
    r35 = fmaf(r21, r81, r35);
    r51 = r2 * r31;
    r52 = r56 + r52;
    r52 = r52 + r67;
    r67 = r30 * r22;
    r56 = r22 * r33;
    r56 = fmaf(r27, r56, r76 * r67);
    r56 = r56 + r68;
    r56 = fmaf(r11, r56, r9 * r52);
    r61 = r54 + r61;
    r56 = fmaf(r10, r61, r56);
    r51 = r51 * r56;
    r35 = fmaf(r32, r51, r35);
    r51 = r29 * r35;
    r51 = fmaf(r1, r51, r60 * r26);
    r26 = r39 * r66;
    r51 = fmaf(r50, r26, r51);
    r26 = r39 * r56;
    r26 = fmaf(r50, r26, r80 * r74);
    r81 = r31 * r35;
    r26 = fmaf(r1, r81, r26);
    r81 = r8 * r36;
    r61 = r2 * r7;
    r61 = r61 * r31;
    r61 = fmaf(r32, r61, r21 * r81);
    r81 = r2 * r45;
    r61 = fmaf(r49, r81, r61);
    r61 = fmaf(r8, r62, r61);
    r81 = r29 * r61;
    r10 = r8 * r49;
    r10 = fmaf(r60, r10, r1 * r81);
    r81 = r45 * r39;
    r10 = fmaf(r50, r81, r10);
    r81 = r31 * r61;
    r54 = r7 * r39;
    r54 = fmaf(r50, r54, r1 * r81);
    r54 = fmaf(r8, r74, r54);
    WriteIdx4<1024, float, float, float4>(
        out_pose_jac, 4 * out_pose_jac_num_alloc, global_thread_idx, r51, r26,
        r10, r54);
    r81 = r42 * r39;
    r11 = r47 * r36;
    r52 = r2 * r46;
    r52 = r52 * r31;
    r52 = fmaf(r32, r52, r21 * r11);
    r11 = r2 * r42;
    r52 = fmaf(r49, r11, r52);
    r52 = fmaf(r47, r62, r52);
    r52 = r52 * r1;
    r81 = fmaf(r29, r52, r50 * r81);
    r11 = r47 * r49;
    r81 = fmaf(r60, r11, r81);
    r11 = r46 * r39;
    r11 = fmaf(r50, r11, r47 * r74);
    r11 = fmaf(r31, r52, r11);
    r52 = r41 * r39;
    r9 = r44 * r49;
    r9 = fmaf(r60, r9, r50 * r52);
    r52 = r2 * r40;
    r52 = r52 * r31;
    r60 = r44 * r36;
    r60 = fmaf(r21, r60, r32 * r52);
    r52 = r2 * r41;
    r60 = fmaf(r49, r52, r60);
    r60 = fmaf(r44, r62, r60);
    r62 = r29 * r60;
    r9 = fmaf(r1, r62, r9);
    r62 = r40 * r39;
    r74 = fmaf(r44, r74, r50 * r62);
    r62 = r31 * r60;
    r74 = fmaf(r1, r62, r74);
    WriteIdx4<1024, float, float, float4>(out_pose_jac,
                                          8 * out_pose_jac_num_alloc,
                                          global_thread_idx, r81, r11, r9, r74);
    r62 = r6 * r4;
    r1 = r6 * r5;
    r1 = fmaf(r28, r1, r69 * r62);
    r62 = r6 * r5;
    r52 = r6 * r4;
    r52 = fmaf(r55, r52, r82 * r62);
    r62 = r6 * r5;
    r68 = r6 * r4;
    r68 = fmaf(r51, r68, r26 * r62);
    r62 = r6 * r4;
    r67 = r6 * r5;
    r67 = fmaf(r54, r67, r10 * r62);
    WriteSum4<float, float>((float *)inout_shared, r1, r52, r68, r67);
  };
  FlushSumShared<4, float>(out_pose_njtr, 0 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = r6 * r4;
    r68 = r6 * r5;
    r68 = fmaf(r11, r68, r81 * r67);
    r67 = r6 * r5;
    r52 = r6 * r4;
    r52 = fmaf(r9, r52, r74 * r67);
    WriteSum2<float, float>((float *)inout_shared, r68, r52);
  };
  FlushSumShared<2, float>(out_pose_njtr, 4 * out_pose_njtr_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r52 = fmaf(r69, r69, r28 * r28);
    r68 = fmaf(r55, r55, r82 * r82);
    r67 = fmaf(r26, r26, r51 * r51);
    r1 = fmaf(r54, r54, r10 * r10);
    WriteSum4<float, float>((float *)inout_shared, r52, r68, r67, r1);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r1 = fmaf(r81, r81, r11 * r11);
    r67 = fmaf(r74, r74, r9 * r9);
    WriteSum2<float, float>((float *)inout_shared, r1, r67);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r67 = fmaf(r69, r55, r28 * r82);
    r1 = fmaf(r69, r51, r28 * r26);
    r68 = fmaf(r69, r10, r28 * r54);
    r52 = fmaf(r69, r81, r28 * r11);
    WriteSum4<float, float>((float *)inout_shared, r67, r1, r68, r52);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r69 = fmaf(r69, r9, r28 * r74);
    r28 = fmaf(r55, r51, r82 * r26);
    r52 = fmaf(r55, r10, r82 * r54);
    r68 = fmaf(r55, r81, r82 * r11);
    WriteSum4<float, float>((float *)inout_shared, r69, r28, r52, r68);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r55 = fmaf(r55, r9, r82 * r74);
    r82 = fmaf(r26, r54, r51 * r10);
    r68 = fmaf(r26, r11, r51 * r81);
    r51 = fmaf(r51, r9, r26 * r74);
    WriteSum4<float, float>((float *)inout_shared, r55, r82, r68, r51);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = fmaf(r10, r81, r54 * r11);
    r54 = fmaf(r54, r74, r10 * r9);
    r9 = fmaf(r81, r9, r11 * r74);
    WriteSum3<float, float>((float *)inout_shared, r51, r54, r9);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r9 = r29 * r39;
    r9 = r9 * r37;
    r54 = r31 * r39;
    r54 = r54 * r37;
    r51 = r29 * r34;
    r51 = r51 * r50;
    r81 = r31 * r34;
    r81 = r81 * r50;
    WriteIdx4<1024, float, float, float4>(out_calib_jac,
                                          0 * out_calib_jac_num_alloc,
                                          global_thread_idx, r9, r54, r51, r81);
    r74 = r6 * r4;
    r11 = r6 * r5;
    r10 = r29 * r4;
    r10 = r10 * r37;
    r68 = r31 * r5;
    r68 = r68 * r37;
    r68 = fmaf(r58, r68, r58 * r10);
    r10 = r6 * r29;
    r10 = r10 * r34;
    r10 = r10 * r4;
    r58 = r6 * r31;
    r58 = r58 * r34;
    r58 = r58 * r5;
    r58 = fmaf(r50, r58, r50 * r10);
    WriteSum4<float, float>((float *)inout_shared, r68, r58, r74, r11);
  };
  FlushSumShared<4, float>(out_calib_njtr, 0 * out_calib_njtr_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = r29 * r39;
    r11 = r11 * r39;
    r74 = r39 * r32;
    r74 = r74 * r21;
    r11 = fmaf(r39, r74, r49 * r11);
    r58 = r32 * r21;
    r68 = r0 * r0;
    r10 = r34 * r34;
    r68 = r68 * r10;
    r10 = r29 * r49;
    r10 = fmaf(r68, r10, r68 * r58);
    WriteSum4<float, float>((float *)inout_shared, r11, r10, r23, r23);
  };
  FlushSumShared<4, float>(out_calib_precond_diag,
                           0 * out_calib_precond_diag_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r23 = r0 * r29;
    r23 = r23 * r34;
    r23 = r23 * r39;
    r10 = r0 * r34;
    r10 = fmaf(r74, r10, r49 * r23);
    WriteSum4<float, float>((float *)inout_shared, r10, r9, r54, r51);
  };
  FlushSumShared<4, float>(out_calib_precond_tril,
                           0 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
  if (global_thread_idx < problem_size) {
    r51 = 0.00000000000000000e+00;
    WriteSum2<float, float>((float *)inout_shared, r81, r51);
  };
  FlushSumShared<2, float>(out_calib_precond_tril,
                           4 * out_calib_precond_tril_num_alloc,
                           calib_indices_loc, (float *)inout_shared);
}

void SimpleRadialFixedPointResJac(
    float *pose, unsigned int pose_num_alloc, SharedIndex *pose_indices,
    float *sensor_from_rig, unsigned int sensor_from_rig_num_alloc,
    float *calib, unsigned int calib_num_alloc, SharedIndex *calib_indices,
    float *pixel, unsigned int pixel_num_alloc, float *point,
    unsigned int point_num_alloc, float *out_res,
    unsigned int out_res_num_alloc, float *out_pose_jac,
    unsigned int out_pose_jac_num_alloc, float *const out_pose_njtr,
    unsigned int out_pose_njtr_num_alloc, float *const out_pose_precond_diag,
    unsigned int out_pose_precond_diag_num_alloc,
    float *const out_pose_precond_tril,
    unsigned int out_pose_precond_tril_num_alloc, float *out_calib_jac,
    unsigned int out_calib_jac_num_alloc, float *const out_calib_njtr,
    unsigned int out_calib_njtr_num_alloc, float *const out_calib_precond_diag,
    unsigned int out_calib_precond_diag_num_alloc,
    float *const out_calib_precond_tril,
    unsigned int out_calib_precond_tril_num_alloc, size_t problem_size) {

  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  SimpleRadialFixedPointResJacKernel<<<n_blocks, 1024>>>(
      pose, pose_num_alloc, pose_indices, sensor_from_rig,
      sensor_from_rig_num_alloc, calib, calib_num_alloc, calib_indices, pixel,
      pixel_num_alloc, point, point_num_alloc, out_res, out_res_num_alloc,
      out_pose_jac, out_pose_jac_num_alloc, out_pose_njtr,
      out_pose_njtr_num_alloc, out_pose_precond_diag,
      out_pose_precond_diag_num_alloc, out_pose_precond_tril,
      out_pose_precond_tril_num_alloc, out_calib_jac, out_calib_jac_num_alloc,
      out_calib_njtr, out_calib_njtr_num_alloc, out_calib_precond_diag,
      out_calib_precond_diag_num_alloc, out_calib_precond_tril,
      out_calib_precond_tril_num_alloc, problem_size);
}

} // namespace caspar